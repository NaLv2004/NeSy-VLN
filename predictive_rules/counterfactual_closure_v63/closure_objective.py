"""Categorical two-step supervision; no neural action or reward model.

All transition calls have hard integer premises in their forward pass. During
training a straight-through gradient crosses the predicted memory ID. For the
two-step open-loop observation likelihood, all64 intermediate observation IDs
are explicitly enumerated; no averaged continuous observation is substituted.
"""
import torch
from torch import nn
from torch.nn import functional as F
from model import hard

TRAINABLE = ('definition_rule.', 'definition_generator.', 'memory_predictor.',
             'memory_rule.', 'action_embedding.', 'observation_embedding.')


def configure(model):
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(TRAINABLE))
    # Forward values remain hard IDs. Allow multi-step credit through the
    # predicted categorical memory during training only.
    model.detach_memory_carrier = False
    return [p for p in model.parameters() if p.requires_grad]


def nll(logp, target):
    return -logp.gather(-1, target[..., None]).squeeze(-1)


class ClosureObjective(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, d, pairs=None):
        model = self.model; b = len(d['context']); device = d['context'].device
        context = d['context']
        a = torch.arange(17, device=device)[None].expand(b, -1)
        m = F.one_hot(d['memory'], 32).float()[:, None].expand(-1, 17, -1)
        o = F.one_hot(d['observation'], 64).float()[:, None].expand(-1, 17, -1)
        lm1, lo1 = model.transition(context, m, o, a)
        first_o_loss = nll(lo1, d['first_observation']).mean()
        first_m_loss = nll(lm1.log_softmax(-1), d['first_memory']).mean()

        # First STOP is terminal and never gets a fabricated second action.
        # Each of the32 valid pairs is equally likely during fitting.
        if pairs is None:
            pairs = torch.arange(2, 34, device=device)[None].expand(b, -1)
        assert pairs.min() >= 2 and pairs.max() < 34
        assert d['second_valid'].flatten(1).gather(1, pairs).all()
        parent = pairs // 2; k = pairs.shape[1]
        index_m = parent[..., None].expand(-1, -1, 32)
        pred_m1 = hard(lm1).gather(1, index_m)
        actual_o1 = d['first_observation'].gather(1, parent)
        a2 = d['second_actions'].flatten(1).gather(1, pairs)
        target_o2 = d['second_observation'].flatten(1).gather(1, pairs)
        target_m2 = d['second_memory'].flatten(1).gather(1, pairs)

        # Matches the allowed deployed process: refresh O from the real image,
        # but keep M from the preceding symbolic transition.
        lm2, lo2 = model.transition(context, pred_m1,
            F.one_hot(actual_o1, 64).float(), a2)
        conditional_o_loss = nll(lo2, target_o2).mean()
        conditional_m_loss = nll(lm2.log_softmax(-1), target_m2).mean()

        # Open-loop planning has no future image. Sum over every categorical
        # O1 outcome while keeping the predicted M1 hard, as in the planner.
        possible_o = torch.eye(64, device=device)[None, None].expand(b, k, -1, -1)
        lm_open, lo_open = model.transition(context,
            pred_m1[:, :, None].expand(-1, -1, 64, -1), possible_o,
            a2[:, :, None].expand(-1, -1, 64))
        parent_logp = lo1.gather(1, parent[..., None].expand(-1, -1, 64))
        marginal_logp = torch.logsumexp(parent_logp[..., None] + lo_open, dim=2)
        open_o_loss = nll(marginal_logp, target_o2).mean()
        # Memory likelihood is an auxiliary categorical supervision target;
        # inference still exports argmax memory IDs, not soft memory carriers.
        marginal_memory = torch.logsumexp(parent_logp[..., None] + lm_open.log_softmax(-1), dim=2)
        open_m_loss = nll(marginal_memory, target_m2).mean()
        loss = (first_o_loss + first_m_loss + .5 *
                (conditional_o_loss + conditional_m_loss + open_o_loss + open_m_loss))

        with torch.no_grad():
            hit = lo1.argmax(-1) == d['first_observation']
            changed = d['first_observation'] != d['observation'][:, None]
            # Hard memory transition probabilities, for an honest metric of
            # the table actually used at runtime (soft likelihood above is
            # NOT reported as hard-state prediction accuracy).
            hard_memory_distribution = torch.zeros(b, k, 32, device=device)
            hard_memory_distribution.scatter_add_(2, lm_open.argmax(-1), parent_logp.exp())
            metrics = dict(loss=loss, first_o_nll=first_o_loss,
                first_m_nll=first_m_loss, conditional_o_nll=conditional_o_loss,
                conditional_m_nll=conditional_m_loss, open_o_nll=open_o_loss,
                open_m_nll=open_m_loss, first_o_hits=hit.sum(),
                first_o_count=hit.numel(), changed_hits=(hit & changed).sum(),
                changed_count=changed.sum(), first_m_hits=(lm1.argmax(-1)==d['first_memory']).sum(),
                conditional_o_hits=(lo2.argmax(-1)==target_o2).sum(),
                conditional_m_hits=(lm2.argmax(-1)==target_m2).sum(),
                open_o_hits=(marginal_logp.argmax(-1)==target_o2).sum(),
                open_hard_m_hits=(hard_memory_distribution.argmax(-1)==target_m2).sum(),
                second_count=target_o2.numel(), positions=b)
        return loss, {name: torch.as_tensor(value, device=device).detach().float() for name,value in metrics.items()}
