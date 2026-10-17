import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules';e=root/'evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
progress=root/'goal_progress.json';p=read(progress)
final=read(e/'discrete_rollout_world_v16_final.json')
assert not final['process_alive'] and final['status']['state']=='training_budget_complete' and final['committed_hash_verified']
record=p['active_runs']['discrete_rollout_world_v16'];record.update(last_observed_state='training_budget_complete',last_observed_update=600,
    checkpoint=final['checkpoint'],validation=final['validation'],recent=final['recent'],committed_hash_verified=True,process_alive=False)
p['local_visual_program_v17_audit']=read(e/'local_visual_program_v17_audit.json')
p['local_visual_program_v17_audit'].update(promoted=False,goal_pointer_trained=False,
    decision='Rejected before goal-pointer fitting or navigation deployment: an oracle future goal still yields only9/64 held first-action matches and34/466 propagated-state matches.')
p['prequant_information_v18']=read(e/'prequant_information_v18.json')
p['prequant_information_v18']['decision']='Supports a bounded complete navigation trial with richer boundary-only visual inputs; absolute movement accuracy is low and both probes overfit. Not a navigation result.'
startup=read(e/'raw_boundary_v19_startup.json');assert startup['process_alive'] and startup['process_identity_verified'] and startup['committed_hash_verified']
p['remote_run']='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_raw_boundary_v19';p['remote_source']=startup['launcher']['source']
p['active_scope']='Complete exact-symbolic navigation candidate with pre-quantization boundary evidence'
p['last_verified_pid']=startup['status']['pid'];p['last_verified_update']=startup['status']['update']
p['last_remote_check_unix']=startup['time'];p['next_regular_check_not_before_unix']=startup['time']+660
p['active_runs']['raw_boundary_v19']=dict(path=p['remote_run'],source=p['remote_source'],last_observed_state='training',
    last_observed_update=startup['status']['update'],pid=startup['status']['pid'],launcher_pid=startup['launcher']['pid'],
    process_identity_verified=True,actual_optimizer_updates_confirmed=True,committed_hash_verified=True,
    checkpoint=startup['checkpoint'],recent=startup['recent'],preflight=read(e/'raw_boundary_v19_preflight.json'),
    initial_checkpoint='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_definition_world_v12/checkpoint_001600.pt',
    development_results=startup['validation'],evaluation_error=startup['evaluation_error'])
p['goal_status']='active';p['completion_proven']=False
p['previous_turn_classification']='progress: component completion and diagnostic evidence; complete raw-boundary navigation candidate launched and verified updating'
p['pending']=[
    'Check the existing v19 launcher only after the recorded660-second interval; verify actual update, committed hash, evaluation process/result and four authorized GPUs. A transport timeout never authorizes restart.',
    'v19 runs600 updates on GPUs3/4/5, development80 every200 on GPU7. It is a complete exact-goal navigation candidate, not the v16 component. Inspect actual navigation success before retaining or extending it.',
    'v19 preserves all37 old grounding/definition tensors. Rich49x128 visual features condition full rule/goal generation only at0/10/20/30; no continuous dynamics targets or within-block neural correction.',
    'Raw transition-bank context is used only with exact causal episode/image correspondence. Monitor counterfactual_raw_boundary_available; unavailable bank rows deliberately retain the original coarse input.',
    'Do not revive rejected v13/v14/v17 candidates. V17 full rule reachability was not navigation correctness; its untrained goal pointer is not deployed.',
    'Keep all checkpoints, training layouts100..115/118..121, development116/117/122/123, and final fixed200..207 isolated. No final fixed-suite tuning.',
    'Reference remains43/80 development44/80 fixed. Last best complete new design is v12step1600=5/80. Component patch accuracy and offline action matching are not success rates.',
    'Only use authorized free GPUs3/4/5/7 and preserve other jobs0/1/2/6. No maps, manual semantic/collision states, value/reward tables, direct actors, approximate proofs or goal replacement.',
    'Navigation performance parity and full end-to-end speed parity remain unproven. Keep the full goal active.'
]
progress.write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(goal_complete=False,current_run=p['remote_run'],verified_update=p['last_verified_update'],
                     next_check_not_before=p['next_regular_check_not_before_unix'])))
