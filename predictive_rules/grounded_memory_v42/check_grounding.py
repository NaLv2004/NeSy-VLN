"""Recorded training batch: independently grounded labels and causal rules."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from model import FactoredWorld,Config,hard
from objectives import objective
from planner import Program

torch.set_num_threads(3)
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
saved=torch.load(root/'onpolicy_successors_v40/policy.pt',map_location='cpu',weights_only=False)
model=FactoredWorld(Config(**saved['model_config']))
model.initialize_v41(saved['model']);c=model.cfg
assert model.memory.weight_ih.data_ptr()!=model.memory_predictor.weight_ih.data_ptr()

torch.manual_seed(42)
ob=F.one_hot(torch.randint(c.observations,(2,6)),c.observations).float()
actions=torch.randint(c.actions,(2,5))
previous=torch.cat([torch.full((2,1),17),actions],1)
evidence=F.normalize(torch.randn(2,6,c.width),dim=-1)
grounded,_,_=model.memory_sequence(ob,previous,evidence)
changed=ob.clone();changed[:,3]=changed[:,3].roll(1,-1)
altered=evidence.clone();altered[:,3]=altered[:,3].roll(1,-1)
modified,_,_=model.memory_sequence(changed,previous,altered)
assert torch.equal(grounded[:,:3],modified[:,:3]), 'Future evidence affected an earlier grounded state'

memory=F.one_hot(torch.arange(c.memories).repeat_interleave(c.observations),c.memories).float()
future=F.one_hot(torch.arange(c.observations).repeat(c.memories),c.observations).float()
action=torch.ones(len(memory),dtype=torch.long)
future_evidence=F.normalize(model.observation_definitions,dim=-1).repeat(c.memories,1)
with torch.no_grad():
    target_before=model.memory_step(memory,future,action,future_evidence)[0].argmax(-1)
    predictor_backup=model.memory_predictor.bias_ih.clone()
    model.memory_predictor.bias_ih.add_(torch.randn_like(predictor_backup)*3)
    target_after=model.memory_step(memory,future,action,future_evidence)[0].argmax(-1)
    assert torch.equal(target_before,target_after), 'Predictor parameters changed their own target labels'
    model.memory_predictor.bias_ih.copy_(predictor_backup)
    variants=[len(row.unique()) for row in target_before.reshape(c.memories,c.observations)]
    assert max(variants)>1,'Grounded target memory ignores all successor evidence'

def forbid_encoder(*_):
    raise RuntimeError('Grounding encoder was called while predicting/compiling rules')
handle=model.memory.register_forward_pre_hook(forbid_encoder)
with torch.no_grad():
    compiled=model.compile(torch.randn(1,c.context),torch.tensor([0]))
    Program(**compiled).validate()
handle.remove()

case=torch.load(root/'run_induced_effects_v39/backward_case_rank0.pt',map_location='cpu',weights_only=False)
data={key:value if key=='operator_targets' else value[:8] for key,value in case['data'].items()}
data['world_updates_encoder']=False
loss,metrics=objective(model,data);loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
result=dict(source_update=saved['update'],independent_predictor_parameters=True,
    future_evidence_cannot_change_earlier_grounded_state=True,
    target_labels_independent_of_predictor=True,target_label_changes_with_future_evidence=True,
    grounded_variants_per_current_memory=variants,no_grounding_encoder_in_rule_compilation=True,
    actual_training_batch_backward_finite=True,loss=float(loss.detach()),
    gradient_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5)),
    single_step_memory_accuracy=float(metrics['single_step_memory_accuracy']),
    branch_memory_accuracy=float(metrics['branch_memory_accuracy']),rollout_memory_accuracy=float(metrics['rollout_memory_accuracy']),
    sampled_training_examples=8,raw_visual_evidence_grounding=True,
    qualification='Numerical/causal checks on training data; not navigation performance or proof of improved representations.')
Path(__file__).with_name('grounding_contracts.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
