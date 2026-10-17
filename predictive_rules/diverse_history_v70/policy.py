from PIL import Image
import torch
from torch.nn import functional as F
from model import hard
from planner import Program,plan,audit

class Policy:
    """Causal history, hard memory, rule renewal, and pure table derivation."""
    def __init__(self, model, task, encoder, gate):
        self.model, self.encoder, self.gate = model, encoder, gate
        self.c = model.cfg
        self.device = next(model.parameters()).device
        self.task = torch.tensor([['chair', 'table'].index(task)], device=self.device)
        self.features, self.observations, self.actions = [], [], []
        self.programs, self.steps = [], []
        self.program = None
        self.context = None
        self.grounding_allowed = False
        self.memory = self.predicted_observation = self.age = None

    @torch.no_grad()
    def choose(self, rgb, defer_planning=False):
        step = len(self.actions)
        with self.gate.boundary(step):
            _, raw = self.encoder.encode_joint([Image.fromarray(rgb)])
            observed_global = int(self.model.observe(raw).argmax(-1))
            observed = int(self.model.observe(raw,self.context).argmax(-1))
            self.features.append(raw[0])
            self.observations.append(observed_global)
            renew, probability = step == 0, 1.
            if step:
                memory = F.one_hot(torch.tensor([self.memory], device=self.device), self.c.memories).float()
                obs = F.one_hot(torch.tensor([observed], device=self.device), self.c.observations).float()
                predicted = F.one_hot(torch.tensor([self.predicted_observation], device=self.device), self.c.observations).float()
                probability = float(self.model.renewal_logits(memory, obs, predicted,
                    torch.tensor([self.age], device=self.device)).sigmoid())
                renew = self.age >= 5 and probability >= .5
            if renew:
                history = torch.stack(self.features)[None]
                previous = torch.tensor([[17] + self.actions], device=self.device)
                observations = F.one_hot(torch.tensor([self.observations], device=self.device), self.c.observations).float()
                context = self.model.context_sequence(history, observations, previous, self.task)[:, -1]
                self.context = context
                local_logits,evidence,_ = self.model.observation_encoding(history,context)
                observations = hard(local_logits)
                observed = int(local_logits[0,-1].argmax(-1))
                self.grounding_allowed = True
                try:
                    memory_logits, _, _ = self.model.memory_sequence(observations, previous,evidence)
                finally:
                    self.grounding_allowed = False
                self.memory = int(memory_logits[0, -1].argmax(-1))
                self.program = Program(**self.model.compile(context, self.task), observations=self.c.observations).validate()
                self.age = 0
                self.programs.append(dict(start=step, goal=self.program.goal, sha256=self.program.digest()))
        prepared = (observed, renew, probability)
        raw = raw[0].cpu().numpy()
        if defer_planning:
            return prepared, raw
        return self.complete_choice(prepared, raw)

    def complete_choice(self, prepared, raw):
        # Pure NumPy rule-table planning; independent environments may run
        # this method concurrently. It never touches a neural module or CUDA.
        observed, renew, probability = prepared
        step = len(self.actions)
        proof = plan(self.program, self.memory, observed, horizon=32-step)
        if not proof['found']:
            self.failed_proof = proof
            return None, raw
        audit(self.program, proof)
        action = int(proof['actions'][0])
        self.steps.append(dict(step=step, observed=observed, memory=self.memory, action=action,
            renewal_probability=probability, renewed=renew, proof=proof))
        # The predicted hard memory is fixed BEFORE executing the action.
        self.memory = int(proof['next_memory'])
        self.predicted_observation = int(proof['predicted_observation'])
        self.age += 1
        return action, raw

    def commit_execution(self, action):
        assert action == self.steps[-1]['action']
        self.actions.append(int(action))



