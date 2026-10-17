import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules';e=root/'evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
p=read(root/'goal_progress.json');final=read(e/'separated_goal_v21_completed.json')
assert not final['process_alive'] and final['committed_hash_verified'] and final['status']['state']=='training_budget_complete'
run=p['active_runs']['separated_goal_v21'];run.update(last_observed_state='training_budget_complete',last_observed_update=1000,
    process_alive=False,checkpoint=final['checkpoint'],development_results=final['validation'],recent=final['recent'],
    decision='Not promoted: development4/80 then3/80 despite improved supervised training-state predictions.')
p['separated_goal_derivations_v21']={k:v for k,v in read(e/'separated_goal_derivations_v21.json').items() if k!='rows'}
seen=read(e/'seen_training_navigation_v21.json');assert seen['episodes']==32 and seen['training_only'] and seen['neural_calls_outside_boundaries']==0
p['seen_training_navigation_v21']=seen
last=read(e/'seen_training_navigation_v21_status.json');assert last['state']=='complete'
p.update(last_verified_pid=None,last_verified_update=1000,last_remote_check_unix=last['time'],next_regular_check_not_before_unix=last['time']+660,
    active_scope='Diagnose remaining closed-loop control gap after completed exact-goal world/goal separation',goal_status='active',completion_proven=False)
p['pending']=[
    'V21 and its training-only32-task audit are complete. No owned training/collector remains active; do not restart terminal runs or rerun their completed development suites.',
    'V21dev500=4/80,1000=3/80. Supervised longer-train-window one-step827/1033 and autonomous642/1033 improved, but seen-task closed-loop is only8/32 (near6/10,medium2/10,far0/12;18dense14normal). Not just unseen-scene generalization.',
    'Next useful evidence: audit actual saved v21 closed-loop programs against replayed visual state labels on these32 training trajectories, especially first rule/state divergence and false STOP. Offline replay labels may not affect the saved executed plans.',
    'The v21 audit folder is /home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/diagnostics/seen_training_navigation_v21; each episode has trace.json, program_*.npz, first/last images. Checkpoint is run_separated_goal_v21/checkpoint_001000.pt,sha a73a308af9f2158e3ca2deef71ee9976a3e237f9deb2dd64dc74686983cad417.',
    'All8066 transition-bank boundary histories were recovered exactly; do not repeat the completed replay. Data: replayed_boundary_features_v20/manifest.json,100%canonical395234patch IDs.',
    'Do not attribute failure to precision or universal gradient conflict: precision audit showed similar accuracy and gradient audit had mixed weak alignment. Oracle final-goal constraints alone did not improve teacher action matching.',
    'All saved checkpoints and partitions remain unchanged. Training layouts100..115/118..121; dev116/117/122/123; finalfixed200..207 remains untouched. Preserve other GPU0/1/2/6 processes; only free3/4/5/7 authorized.',
    'Network-generated full anonymous definitions/rules/discrete goal, immutable exact full proofs, no learned reward/value table, semantic state, map, neural actor, goal replacement or intermediate observation correction.',
    'Best new exact-symbolic development result remainsv12step1600=5/80, reference43/80development44/80fixed. Goal remains active; neither full navigation nor full execution-speed parity is established.'
]
p['previous_turn_classification']='progress: two full candidates completed; canonical transition inputs recovered; learned dynamics improved; train-only closed-loop gap measured; goal unachieved'
(root/'goal_progress.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(goal_complete=False,latest_update=1000,development=[(r['update'],r['successes']) for r in final['validation']],seen_training=[seen['successes'],seen['episodes']],owned_training_active=False)))
