"""Record component evidence separately from navigation performance."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules';e=root/'evidence'
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
path=root/'goal_progress.json';p=read(path)
startup=read(e/'discrete_rollout_world_v16_startup.json')
assert startup['process_identity_verified'] and startup['committed_hash_verified']
p['last_navigation_run']=p['remote_run'];p['last_navigation_source']=p['remote_source']
p['remote_run']='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_discrete_rollout_world_v16'
p['remote_source']=startup['launcher']['source'];p['active_scope']='Dynamics component only; full navigation contract and parity remain required'
p['last_verified_pid']=startup['status']['pid'];p['last_verified_update']=startup['status']['update']
p['last_remote_check_unix']=startup['time'];p['next_regular_check_not_before_unix']=startup['time']+660
p['completion_proven']=False;p['goal_status']='active'
p['transition_information_audit_v12']=read(e/'transition_information_v12.json')
p['transition_information_audit_v12']['conclusion']='Visual pre-quantization features only modestly improve held movement prediction; quantization alone is not established as the dominant bottleneck.'
p['soft_categorical_dynamics_v14']=read(e/'soft_symbol_dynamics_v14.json')
p['soft_categorical_dynamics_v14'].update(promoted=False,decision='Lower posterior KL did not improve held movement-state accuracy; no navigation rollout was run.')
p['active_runs']['discrete_visual_world_v15']=dict(
    path='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_discrete_visual_world_v15',
    scope='component_only',last_observed_state='training_budget_complete',last_observed_update=800,
    terminal=read(e/'discrete_visual_world_v15_terminal.json'),latest_commit=read(e/'discrete_visual_world_v15_final_commit.json'),
    validation=[json.loads(line) for line in (e/'discrete_visual_world_v15_validation.jsonl').read_text().splitlines()],
    navigation_success_rate=None,promoted_to_controller=False)
p['discrete_visual_world_v15_autonomous_audit']=read(e/'discrete_world_600_rollout_v15.json')
p['active_runs']['discrete_rollout_world_v16']=dict(path=p['remote_run'],source=p['remote_source'],
    scope='component_only',last_observed_state=startup['status']['state'],last_observed_update=startup['status']['update'],
    pid=startup['status']['pid'],launcher_pid=startup['launcher']['pid'],actual_optimizer_updates_confirmed=True,
    committed_hash_verified=True,checkpoint=startup['checkpoint'],recent=startup['recent'],validation=startup['validation'],
    baseline=read(e/'discrete_rollout_world_v16_baseline.json'),navigation_success_rate=None,promoted_to_controller=False)
p['pending']=[
    'V16 is a bounded600-update dynamics fit, not a complete navigation model. Check the existing handle after the recorded660-second interval; preserve its checkpoint on any transport error.',
    'On internal held training episodes at v16step400, autonomous tenth-frame patch accuracy rose21.63% to33.20%; changed-patch accuracy16.39% to21.77%. This is not task success, a symbolic proof result or deployment evidence.',
    'Complete boundary-time generation of definitions, full transition rules and an unmodified discrete subgoal is still required for any successor navigation candidate. Do not deploy the neural dynamics component to choose actions between revisions.',
    'Do not introduce maps, semantic/collision/eligibility heads, learned reward/value tables, direct actors or unreachable-goal replacement. All executed actions need a complete exact proof.',
    'Preserve v12best1600=5/80 and final2000=2/80. Reference remains43/80 development and44/80 fixed; do not conflate component metrics with this comparison.',
    'No final fixed-suite tuning or additional evaluations of rejected v13/v14 probes. Training/development/fixed splits remain unchanged.',
    'Only free authorized GPUs3/4/5/7 may be used. Other jobs0/1/2/6 were preserved; never kill them.',
    'Navigation parity and full execution-speed parity remain unproven. The goal must remain active.'
]
path.write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(goal_complete=False,active_run=p['remote_run'],update=p['last_verified_update'],
                     next_regular_check_not_before_unix=p['next_regular_check_not_before_unix'])))
