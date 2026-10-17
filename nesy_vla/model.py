import copy,math
import torch
from torch import nn
from .generator import VLMProgramGenerator
from .perception import PredicateGrounder
from .logic import SoftLogic
from .runtime import image_tensor


class NavigationModel(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg
        if cfg.get('generator_backend')=='task_tokens':
            from .task_token_generator import TaskTokenProgramGenerator
            self.generator=TaskTokenProgramGenerator(cfg['program'],cfg.get('generator_hidden',64),cfg['task_vocabulary'])
        elif cfg.get('generator_backend')=='symbolic_gru':
            from .small_generator import SymbolicProgramGenerator
            self.generator=SymbolicProgramGenerator(cfg['program'],cfg.get('generator_hidden',64))
        else:self.generator=VLMProgramGenerator(cfg['model_path'],cfg['program'])
        backbone=None
        if cfg.get('vision_backbone')=='qwen35_frozen':
            from .vision_backbone import FrozenQwenVision
            backbone=FrozenQwenVision(cfg['model_path'],cfg.get('vision_microbatch',8))
        p=cfg['program']; self.grounder=PredicateGrounder(p['d'],p['a'],cfg['imagenet_weights'],
            cfg.get('evidence_only_grounder',False),cfg.get('spatial_encoding',False),backbone,cfg.get('freeze_vision',False),
            cfg.get('vision_cache_entries',0),cfg.get('canonical_vision_batch',0),cfg.get('vision_calibration'))
        # A frozen visual tower is immutable and shared; only learned predicate
        # grounding parameters require a separate EMA teacher.
        shared={} if any(p.requires_grad for p in self.grounder.vision.parameters()) else {id(self.grounder.vision):self.grounder.vision}
        self.teacher=copy.deepcopy(self.grounder,memo=shared).requires_grad_(False).eval()
        if cfg['program'].get('action_semantics')=='conjunctive_rules':
            if not cfg.get('discrete_reasoning',False):raise ValueError('Conjunction rules require discrete states')
            from .conjunctive_logic import ConjunctiveRuleLogic
            self.logic=ConjunctiveRuleLogic(p['k'],p['a'],p['r'],
                p.get('neural_conjunction_gates',False) or cfg.get('install_neural_conjunction_gates',False))
        elif cfg['program'].get('action_semantics')=='branch_tree':
            if not cfg.get('discrete_reasoning',False):raise ValueError('Branch rules require discrete states')
            from .branch_logic import TotalBranchLogic
            self.logic=TotalBranchLogic(p['k'],p['a'],p['r'])
        elif cfg.get('discrete_reasoning',False):
            from .neural_logic import NeuralSymbolicLogic
            self.logic=NeuralSymbolicLogic(p['k'],p['a'],p['r'])
        else:self.logic=SoftLogic(p['k'],p['a'],p['r'])
        if cfg.get('execution_outcome_supervision',False):
            if not cfg.get('discrete_reasoning',False):raise ValueError('Execution supervision requires discrete symbols')
            from .outcome_supervision import SymbolicExecutionDecoder
            self.execution_decoder=SymbolicExecutionDecoder(p['d'],p['a'])

    def train(self,mode=True):
        super().train(mode); self.teacher.eval()
        # Disable stochastic depth for exact PPO recomputation; gradients remain enabled.
        self.grounder.vision.eval()
        return self

    @property
    def device(self):return next(self.grounder.parameters()).device

    def step(self,program,image,memory,prior=None,tokens=None,replay_perceptual_codes=None):
        if tokens is None:tokens=self.grounder.encode(image_tensor(image,self.cfg['image_size'],self.device))
        observation,confidence=self.grounder.observe(tokens,program.definitions,memory)
        if self.cfg.get('discrete_reasoning',False):
            from .symbols import tokenize
            # The perceptual history stays inside grounding. Neither image
            # features nor visual confidence reaches the rule interpreter.
            observed=tokenize(observation);fresh_codes=observed.codes
            if replay_perceptual_codes is not None:
                from .symbols import replay_observed_tokens
                observed=replay_observed_tokens(observed,replay_perceptual_codes)
            predicted=None if prior is None else tokenize(prior[0])
            state=self.logic.assimilate(observed,predicted)
            proof=self.logic.infer(program,state)
            proof['perceptual_codes']=observed.codes
            proof['fresh_perceptual_codes']=fresh_codes
            proof['prior_codes']=None if predicted is None else predicted.codes
            value,_=self.grounder.readout(program.definitions,proof['belief'],proof['confidence'])
            return dict(proof,tokens=tokens,observation=observation,observed_confidence=confidence,value=value)
        if prior is None:belief=observation; certainty=confidence
        else:
            predicted,prior_c=prior
            belief=(1-confidence)*predicted+confidence*observation
            certainty=1-(1-prior_c)*(1-confidence)
        proof=self.logic.infer(program,belief,certainty)
        value,_=self.grounder.readout(program.definitions,proof['belief'],proof['confidence'])
        return dict(proof,tokens=tokens,observation=observation,observed_confidence=confidence,value=value)

    def advance(self,program,result,action,feedback,memory):
        retention=self.grounder.retain(program.definitions,action)
        b,c,conflict=self.logic.transition(program,result['belief'],result['confidence'],action,retention)
        event=self.grounder.remember(result['tokens'],action,
            torch.tensor(feedback,device=self.device,dtype=torch.float32))
        return (b,c),[*memory[-15:],event],conflict

    @torch.no_grad()
    def update_teacher(self,decay=None):
        decay=self.cfg.get('teacher_decay',.995) if decay is None else decay
        if not math.isfinite(decay) or not 0<=decay<1:
            raise ValueError('Teacher decay must be finite and in [0, 1)')
        for a,b in zip(self.teacher.parameters(),self.grounder.parameters()):
            if a is not b:a.lerp_(b,1-decay)

    def transition_target(self,program,image,memory,tokens=None):
        with torch.no_grad():
            if tokens is None:tokens=self.teacher.encode(image_tensor(image,self.cfg['image_size'],self.device))
            # The target must not reveal the action it is supposed to explain.
            evidence=[] if self.cfg.get('visual_transition_target',False) else [v.detach() for v in memory]
            observation,confidence=self.teacher.observe(tokens,program.definitions.detach(),evidence)
            if self.cfg.get('discrete_reasoning',False):
                from .symbols import tokenize
                observation=tokenize(observation).values;confidence=torch.ones_like(observation)
            return observation,confidence
