"""Generated higher-order conjunctions over anonymous finite state IDs.

This augments the existing learned pairwise clauses with product-t-norm
clauses that may reference every learned state factor. No visual latent
is an argument of the rule interpreter.
"""
import math
from dataclasses import dataclass
import torch
from torch import nn
from .factorized_automaton import AutomatonProgram
from .probabilistic_automaton import ProbabilisticTokenAutomaton


@dataclass
class HigherOrderProgram(AutomatonProgram):
    premise: torch.Tensor | None = None
    unused: torch.Tensor | None = None
    higher_conclusions: torch.Tensor | None = None
    higher_prior: torch.Tensor | None = None
    higher_mix: torch.Tensor | None = None


class HigherOrderAutomaton(ProbabilisticTokenAutomaton):
    def __init__(self, higher_rules=128, **kwargs):
        super().__init__(**kwargs)
        if not self.task_banks:
            raise ValueError('This experiment requires learned goal-indexed programs')
        self.higher_rules = higher_rules
        self.higher_premise = nn.Parameter(torch.randn(self.tasks, higher_rules, self.factors, self.states) * .2)
        self.higher_premise_basis = nn.Parameter(torch.randn(self.rank, higher_rules, self.factors, self.states) * .01)
        self.higher_unused = nn.Parameter(torch.zeros(self.tasks, higher_rules, self.factors))
        self.higher_unused_basis = nn.Parameter(torch.randn(self.rank, higher_rules, self.factors) * .01)
        self.higher_conclusions = nn.Parameter(torch.randn(self.tasks, higher_rules, self.actions) * .03)
        self.higher_conclusion_basis = nn.Parameter(torch.randn(self.rank, higher_rules, self.actions) * .01)
        self.higher_prior = nn.Parameter(torch.zeros(self.tasks, higher_rules))
        self.higher_prior_basis = nn.Parameter(torch.randn(self.rank, higher_rules) * .01)
        self.higher_mix = nn.Parameter(torch.full((self.tasks,), math.log(.1 / .9)))

    def initialize_pairwise(self, state):
        target = self.state_dict()
        assert set(state) == {k for k in target if not k.startswith('higher_')}
        target.update(state); self.load_state_dict(target)

    def program_from_coefficients(self, c, start, tasks=None):
        base = super().program_from_coefficients(c, start, tasks)
        return HigherOrderProgram(**vars(base),
            premise=(self.higher_premise[tasks] + torch.einsum('br,rlfs->blfs', c, self.higher_premise_basis)).softmax(-1),
            unused=(self.higher_unused[tasks] + torch.einsum('br,rlf->blf', c, self.higher_unused_basis)).sigmoid(),
            higher_conclusions=(self.higher_conclusions[tasks] + torch.einsum('br,rla->bla', c, self.higher_conclusion_basis)).softmax(-1),
            higher_prior=(self.higher_prior[tasks] + c @ self.higher_prior_basis).softmax(-1),
            higher_mix=self.higher_mix[tasks].sigmoid())

    def higher_rule_proof(self, program, state_values, step):
        if not program.start <= step < program.start + 10:
            raise ValueError('Expired program')
        self.validate_belief(state_values, len(program.coefficients))
        # A premise is a distribution over categorical equalities. The learned
        # unused branch is logical TRUE; no factor is assigned a human meaning.
        equality = torch.einsum('bfs,blfs->blf', state_values, program.premise)
        factor_support = program.unused + (1 - program.unused) * equality
        conjunction = factor_support.prod(-1)
        support = program.higher_prior * conjunction
        posterior = support / support.sum(-1, keepdim=True).clamp_min(1e-30)
        probabilities = torch.einsum('bl,bla->ba', posterior, program.higher_conclusions)
        return dict(probabilities=probabilities, factor_support=factor_support,
                    conjunction=conjunction, normalized_rule_support=posterior)

    def apply_action_rules(self, program, state_values, step):
        pair, pair_evidence = super().apply_action_rules(program, state_values, step)
        higher = self.higher_rule_proof(program, state_values, step)
        mix = program.higher_mix[:, None]
        # Both branches are explicit soft rule interpreters over state IDs.
        return (1 - mix) * pair + mix * higher['probabilities'], torch.cat([pair_evidence, higher['probabilities'][:, None]], 1)

    @torch.no_grad()
    def proof(self, program, observed_ids, previous_belief, previous_action, state_belief, step):
        result = super().proof(program, observed_ids, previous_belief, previous_action, state_belief, step)
        result['higher_order'] = self.higher_rule_proof(program, result['state_belief'], step)
        return result
