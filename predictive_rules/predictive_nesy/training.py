"""Open-loop multi-step supervision of generated symbolic dynamics."""
import torch
import torch.nn.functional as F
from .model import straight_through, transition_all, read_scalar, risk_all, one_step_scores


def weighted_mean(values, valid):
    return (values * valid).sum() / valid.sum().clamp_min(1)


def objective(model, batch):
    starts = batch["start"]
    if not torch.all(starts == starts[0]):
        raise ValueError("Batch together the same revision boundary")
    start = int(starts[0])
    width = min(start + 1, model.cfg.history)
    program = model.generate(batch["task"], batch["history"][:, :width],
        batch["previous_actions"][:, :width], batch["history_length"], start=start)
    # These future groundings are targets/auxiliary supervision. They are never
    # fed into the rollout state after its initialization below.
    observed = model.ground(program, batch["targets"])
    hard_observed = straight_through(observed)
    valid = batch["valid"].float()
    b, t, f, k = observed.shape
    teacher = batch["teacher_ids"]
    ground_ce = -observed.clamp_min(1e-8).log().gather(-1, teacher[..., None]).squeeze(-1).mean(-1)
    reconstruction = model.reconstruct(hard_observed.flatten(-2)).reshape(b, t, model.cfg.patches, model.cfg.vocabulary)
    recon_ce = F.cross_entropy(reconstruction.flatten(0, 2), batch["targets"].flatten(), reduction="none").reshape(b, t, -1).mean(-1)
    # Immutable teacher labels come from the previously learned perception,
    # without handcrafted predicate meanings or future image features in rules.
    # Current definitions are trained to ground the same learned alphabet.
    losses = {"grounding": weighted_mean(ground_ce, valid), "reconstruction": weighted_mean(recon_ce, valid)}
    state = hard_observed[:, 0]
    dyn_losses, dyn_valid, action_losses, action_valid = [], [], [], []
    goal_losses, value_losses, risk_losses = [], [], []
    symbol_correct, action_correct = [], []
    for step in range(t):
        # This is the sole state path: every previous prediction remains in it.
        q = one_step_scores(program, state, model.pairs)
        labels = batch["actions"][:, step]
        action_losses.append(F.cross_entropy(q / .1, labels, reduction="none"))
        action_valid.append(valid[:, step] * batch["supervised"][:, step])
        action_correct.append((q.argmax(-1) == labels).float())
        goal = read_scalar(program, state, "goals", model.pairs)
        value = read_scalar(program, state, "values", model.pairs)
        goal=goal.float().clamp(1e-6,1-1e-6)
        goal_label=batch['goal'][:,step].float()
        goal_losses.append(-goal_label*goal.log()-(1-goal_label)*torch.log1p(-goal))
        value_losses.append((value - batch["value"][:, step]).square())
        predicted_risk = risk_all(program, state).float().clamp(1e-6, 1-1e-6)
        measured = batch['risks'][:, step]
        risk_label=measured.clamp_min(0)
        risk_losses.append(-risk_label*predicted_risk.log()-(1-risk_label)*torch.log1p(-predicted_risk))
        symbol_correct.append((state.argmax(-1) == teacher[:, step]).float().mean(-1))
        if step < t - 1:
            probabilities = transition_all(program, state)[torch.arange(b, device=state.device), labels]
            dyn_losses.append(-probabilities.clamp_min(1e-8).log().gather(-1, teacher[:, step + 1, :, None]).squeeze(-1).mean(-1))
            dyn_valid.append(valid[:, step] * valid[:, step + 1] * (labels != 0))
            state = straight_through(probabilities)
    amask = torch.stack(action_valid, 1)
    losses["planning"] = weighted_mean(torch.stack(action_losses, 1), amask)
    losses["dynamics"] = weighted_mean(torch.stack(dyn_losses, 1), torch.stack(dyn_valid, 1))
    losses["goal"] = weighted_mean(torch.stack(goal_losses, 1), valid)
    losses["value"] = weighted_mean(torch.stack(value_losses, 1), valid)
    losses["risk"] = weighted_mean(torch.stack(risk_losses, 1), (batch['risks'] >= 0).float() * valid[..., None])
    inverse = model.inverse(torch.cat([hard_observed[:, :-1].flatten(-2), hard_observed[:, 1:].flatten(-2)], -1))
    inverse_loss = F.cross_entropy(inverse.flatten(0, 1), batch["actions"][:, :-1].flatten(), reduction="none").reshape(b, t-1)
    losses["inverse"] = weighted_mean(inverse_loss, valid[:, 1:] * valid[:, :-1])
    mean = (observed * valid[:, :, None, None]).sum((0, 1)) / valid.sum().clamp_min(1)
    losses["balance"] = (mean * (mean.clamp_min(1e-8) * k).log()).sum(-1).mean()
    weights = dict(grounding=1., reconstruction=.1, planning=1., dynamics=1., goal=1., value=2., risk=.1, inverse=.1, balance=.01)
    total = sum(weights[name] * loss for name, loss in losses.items())
    metrics = {name: float(value.detach()) for name, value in losses.items()}
    metrics["loss"] = float(total.detach())
    metrics["open_loop_action_accuracy"] = float(weighted_mean(torch.stack(action_correct, 1), amask).detach())
    metrics["open_loop_state_accuracy"] = float(weighted_mean(torch.stack(symbol_correct, 1), valid).detach())
    metrics["state_accuracy_t10"] = float(weighted_mean(symbol_correct[-1], valid[:, -1]).detach())
    metrics["grounding_accuracy"] = float(weighted_mean((observed.argmax(-1) == teacher).float().mean(-1), valid).detach())
    return total, metrics
