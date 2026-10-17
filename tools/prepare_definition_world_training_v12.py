"""Prepare new named training entries; preserve every previously launched source."""
from pathlib import Path
from predictive_remote import connect
local=Path(__file__).resolve().parents[1]/'predictive_rules'
root='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015';source=root+'/src_definition_world_v12'

def make(old,new,replacements):
    target=local/new
    if target.exists():raise RuntimeError('Named entry already exists; inspect instead of overwriting: '+new)
    value=(local/old).read_text(encoding='utf-8')
    for before,after in replacements:
        assert before in value,before
        value=value.replace(before,after)
    target.write_text(value,encoding='utf-8');return target

model_import=('from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel',
    'from predictive_nesy.definition_conditioned_world import DefinitionConditionedWorld')
trainer=make('train_vocabulary_rules_v11.py','train_definition_world_v12.py',[
    model_import,('anonymous_rollout_proof_subgoal_v8','anonymous_definition_conditioned_world_v12'),
    ("('transitions.','subgoal.')","('definition_rules.','subgoal.')"),
    ("['transitions','subgoal']","['definition_rules','subgoal']"),
    ('evaluate_rollout_proof.py','evaluate_definition_world_v12.py'),
    ("initialization_scope='retained learned image/history features, full neural definitions distilled from unsupervised visual clusters, fresh rule and goal heads'",
     "initialization_scope='complete v11 representation and goal head; new definition-conditioned rule decoder initialized by training-only distillation'"),
    ("definition_initialization=old['definition_initialization'],","definition_initialization=old['definition_initialization'], definition_rule_initialization=old['definition_rule_initialization'],"),
    ("world_model_intervention='stable automatically learned visual vocabulary, no semantic state labels or runtime clustering file'",
     "world_model_intervention='complete transition rules conditioned on learned source and destination definition vectors; no manually assigned semantics'")])
evaluation=make('evaluate_rollout_proof.py','evaluate_definition_world_v12.py',[
    model_import,('anonymous_rollout_proof_subgoal_v8','anonymous_definition_conditioned_world_v12')])
preflight=make('preflight_vocabulary_rules.py','preflight_definition_world_v12.py',[
    model_import,('VisualMemorySubgoalModel(','DefinitionConditionedWorld('),
    ("initialization_vocabulary_v11r1/report.json').read_text())['passed']","initialization_definition_world_v12/report.json').read_text())['complete']"),
    ('initialization_vocabulary_v11r1/initial.pt','initialization_definition_world_v12/initial.pt'),
    ("('transitions.','subgoal.')","('definition_rules.','subgoal.')"),
    ('m.transitions.decoder[-1].weight','m.definition_rules.query[-1].weight'),
    ('counterfactual_proof_v9','counterfactual_proof_v11'),
    ('vocabulary_rules_preflight.json','definition_world_v12_preflight.json')])
c=connect()
try:
    with c.open_sftp() as ftp:
        for path in [trainer,evaluation,preflight]:ftp.put(str(path),source+'/'+path.name)
    print('Prepared new v12 training/evaluation/preflight entries')
finally:c.close()
