"""Soft logical beliefs over learned discrete states, with replayable proofs.

The perceptual boundary remains integer visual tokens, followed by integer
observation-predicate IDs. No visual feature reaches the transition interpreter.
Beliefs are explicit probabilities over a finite set of anonymous state IDs.
"""
import torch
import torch.nn.functional as F
from .factorized_automaton import FactorizedTokenAutomaton, categorical


class ProbabilisticTokenAutomaton(FactorizedTokenAutomaton):
    def validate_belief(self, belief, batch):
        if belief.shape != (batch, self.factors, self.states):
            raise ValueError('Expected a distribution over each discrete state factor')
        if not torch.isfinite(belief).all() or (belief.detach() < -1e-7).any():
            raise ValueError('Invalid symbolic belief probabilities')
        if not torch.allclose(belief.detach().sum(-1), belief.new_ones((batch, self.factors)), atol=2e-5, rtol=0):
            raise ValueError('Symbolic state probabilities must sum to one')

    def transition_belief(self, program, observed, previous, previous_action, step):
        if not program.start <= step < program.start + 10:
            raise ValueError('Expired program')
        self.validate_belief(previous, len(observed))
        # Observation premises are still hard IDs. Straight-through gradients
        # train the predicate definitions, but never change forward premise values.
        if observed.shape != previous.shape or not torch.equal(observed.detach(), observed.detach().round()):
            raise ValueError('Observed predicates must be categorical equalities')
        self.validate_belief(observed, len(observed))
        if previous_action.dtype != torch.long or previous_action.shape != (len(observed),) or ((previous_action < 0) | (previous_action > self.actions)).any():
            raise TypeError('Invalid executed-action tokens')
        if self.reset_memory:
            _, previous = self.initial_state(len(observed), observed.device)
            previous_action = torch.full_like(previous_action, self.actions)
        tables = (self.transition[program.tasks, :, :, :, previous_action, :] if self.task_banks else
                  self.transition[:, :, :, previous_action, :].permute(3, 0, 1, 2, 4))
        tables = (tables + program.transition_delta[:, :, :, None, :]).softmax(-1)
        # Exact marginalization of explicit finite-state transition rules:
        # P(s' = k) = sum_o,p [observed=o] P(s=p) T(k | o,p,executed_action).
        successor = torch.einsum('bfo,bfp,bfops->bfs', observed, previous, tables)
        self.validate_belief(successor, len(observed))
        return successor

    def apply_action_rules(self, program, state_values, step):
        if not program.start <= step < program.start + 10:
            raise ValueError('Expired program')
        self.validate_belief(state_values, len(program.coefficients))
        base = self.conclusions[program.tasks] if self.task_banks else self.conclusions[None]
        tables = (base + program.action_delta).softmax(-1)
        support = []
        for r, (i, j) in enumerate(self.pairs):
            # Product t-norm for conjunction, under the factorized-belief
            # approximation; all possible discrete equality pairs contribute.
            support.append(torch.einsum('bu,bv,buva->ba', state_values[:, i], state_values[:, j], tables[:, r]))
        evidence = torch.stack(support, 1)
        probabilities = (program.rule_weights[:, :, None] * evidence).sum(1)
        return probabilities, evidence

    def step(self, program, codes, previous_values, previous_action, step):
        self.validate_codes(codes)
        if codes.ndim != 2:
            raise ValueError('Expected one integer symbolic observation per trajectory')
        features = self.observation(self.embed(codes)).reshape(-1, self.factors, self.width)
        logits = torch.einsum('bfd,bfkd->bfk', F.normalize(features, dim=-1), F.normalize(program.definitions, dim=-1)) * 5.
        observed_prob = logits.softmax(-1)
        observed_ids, observed = categorical(observed_prob)
        state_prob = self.transition_belief(program, observed, previous_values, previous_action, step)
        probabilities, evidence = self.apply_action_rules(program, state_prob, step)
        return dict(probabilities=probabilities, rule_evidence=evidence, observed_ids=observed_ids,
                    observed_values=observed, observed_probabilities=observed_prob,
                    state_ids=state_prob.argmax(-1), state_values=state_prob,
                    state_probabilities=state_prob,
                    reconstruction=self.reconstruct(observed.flatten(1)).reshape(-1, self.patches, self.vocabulary))

    @torch.no_grad()
    def proof(self, program, observed_ids, previous_belief, previous_action, state_belief, step):
        if observed_ids.dtype != torch.long or observed_ids.shape != (len(program.coefficients), self.factors) or ((observed_ids < 0) | (observed_ids >= self.states)).any():
            raise TypeError('Proof requires integer observation-predicate IDs')
        observed = F.one_hot(observed_ids, self.states).float()
        derived = self.transition_belief(program, observed, previous_belief, previous_action, step)
        self.validate_belief(state_belief, len(observed))
        if not torch.allclose(derived, state_belief, atol=1e-7, rtol=0):
            raise ValueError('Recorded belief has no transition proof')
        probabilities, evidence = self.apply_action_rules(program, derived, step)
        return dict(probabilities=probabilities, rule_evidence=evidence,
                    transition_probabilities=derived, pairs=self.pairs,
                    predicate_ids=observed_ids, previous_belief=previous_belief,
                    state_belief=derived)
