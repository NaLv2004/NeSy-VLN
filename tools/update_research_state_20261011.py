import json,datetime,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'output';path=out/'research_state_20261011.json'
def read(name):return json.loads((out/name).read_text(encoding='utf-8'))
old=read(path.name);backup=out/'research_state_before_v9_v10_20261011.json'
if not backup.exists():shutil.copy2(path,backup)
j=read('joint_v9_status.json');v8=read('frozen_v8_status.json');v7=read('robust_v7_status.json');v10=read('calibrated_v10_status.json')
ground=read('frozen_grounding_audit_step10.json')
record={'goal':'Original isolated fixed80 navigation success >=70%; not achieved.',
    'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'previous_turn_classification':'Explanation-only user turn: no training progress. This continuation made verified code, diagnostic and training progress.',
    'constraints':['No maps or manual predicate semantics','Every action from generated symbolic rules',
        'Neural inference receives discrete symbols only','Visual calibration and feature encoders stay inside predicate grounding',
        'Max32 actions','Program revision every10 actions','Original train/development/fixed80 split unchanged','8 GPUs authorized'],
    'best_observed_fixed80':{'run':'symbolic_robust_v7/runs/robust_multieffect','update':80,'successes':22,'episodes':80,
        'development_successes':7,'collision_action_rate':.5466337245213094,
        'note':'Historical highest observed fixed score, not achieved target. Source selection for joint v9 used development results instead.'},
    'active':{
        'joint_v9':j,
        'robust_multieffect_v7':{'observed_time':v7['time'],**v7['jobs']['robust_multieffect'],'gpus':[3,4,5]},
        'frozen_visual_v8':v8,
        'calibration_pilots_v10':v10},
    'retired_this_turn':{'robust_reference':j['allocation']},
    'older_retired':old.get('retired',{}),
    'v9_changes':{'initialization':'v7 multieffect best_development step40, full model+optimizer',
        'source_sha256':'7c4c0e80a17c094883eff1adcc18fa96596763d05194bed093fdd17ed3bb7960',
        'training':'24 stochastic rollouts and36 successful demonstrations per update across GPUs0,1,2',
        'inverse':'Compare hard predicted future symbols, not subthreshold visual confidence',
        'ppo_fix':'Canonical vision batch8 and replay of actually observed Boolean tokens with straight-through gradients',
        'measured_update2':'Three ranks PPO ratio=1 and fresh bit mismatch=0; sampled training success6/24; about27.74seconds/update',
        'latest_verified':'Update14, three live ranks; about21.89seconds/update; PPO ratio1 and fresh bit mismatch0 persist.',
        'step10_validation':'Development4/80; fixed10/80 (chairs2/40, tables8/40). Not a meaningful improvement over initialization development4/80.',
        'failure_shift':'Same development80: false STOP failures29->56, horizon failures47->20; success unchanged4. Lower collision rate is partly shorter episodes, not proven improved navigation.',
        'traces':'One rotating full rollout per rank/update; all episode IDs and outcomes logged; every validation trajectory retained'},
    'v10_changes':{'basis':'Qwen visual features had high common channel bias and nearly uniform spatial attention',
        'calibration':'Per-channel mean and scale from512 locked-training images, without labels',
        'controlled_comparison':'Both new pilots restore identical symbolic source and train48 demonstrations/update for10updates, only calibration differs',
        'first_update':'Predicate std .0290 control vs .1023 calibrated; navigation improvement unproven; augmentation disagreement increased too',
        'update10':'Both completed ten updates, actual processes still evaluating at inspection. Control predicate std .0132, calibrated .0948; expert-action agreement .327 vs .375; no completed navigation validation yet.',
        'tests':59,'gpus':'6 and7, each pilot limited to20%memory, sharing with v8'},
    'grounding_audit':{'source':'frozen_grounding_audit_step10.json','frames':len(ground['frames']),
        'frozen_convnext':{k:{a:b for a,b in v.items() if a not in ['truths','actions','bit_frequency']} for k,v in ground['results'][0]['goals'].items()},
        'frozen_qwen':{k:{a:b for a,b in v.items() if a not in ['truths','actions','bit_frequency']} for k,v in ground['results'][1]['goals'].items()},
        'qualification':'Training images only, fixed program and empty memory isolate grounding; not navigation success.'},
    'additional_evidence':['inverse_symbol_codes.json','development30_symbolic_audit.json','ppo_replay_audit_v9.json','discrete_information_probe_summary.json','goal_conditioning_audit_v9.json','training_goal_pair_audit.json','joint_v9_development10_failure_audit.json','joint_v9_training_return_audit.json'],
    'goal_conditioning_audit':'Changing chair to table yields ~.002 RMS definition changes, tiny initial rule-routing differences. Investigate goal dependence; this alone does not prove behavior ignores goals.',
    'training_goal_pairs':'17560 training PNGs contain zero exactly identical cross-goal image pairs. This does not prove absence of semantically similar scenes.',
    'reward_audit':'57 rotating saved training trajectories:3 successes mean return30.65;35 false STOP failures mean-16.32;19 horizon failures mean-43.13. All19 discounted horizon returns below immediate false-STOP reference-12.04. Not an executed counterfactual; inspect reward incentive before changing it.',
    'storage':{'completed_evaluations':'v7 older full traces archived losslessly under evidence_archives/completed_evaluations, with per-file hashes; summaries and checkpoints retained',
        'cold_checkpoint_compression':'32 historical latest snapshots first compressed losslessly, freeing .872GiB. All model weights preserved.',
        'optimizer_cleanup':'31 retired latest checkpoints then compacted by removing unused optimizer slots. Every non-optimizer field verified equal; best model plus full optimizer retained for every run. Freed additional5.155GiB.',
        'reader':'stripsnav.io.load_committed reads remaining legacy cold checkpoint and verifies decompressed SHA256 without disk rehydration; verified on a real archived checkpoint.',
        'manifest':'/home/user/xjy/strips_wm_navigation_20261009/retired_optimizer_compaction.json',
        'capacity_warning':'After compaction free space7.24GiB, then6.14GiB at latest training check. Continue tracking growth; current NeSy checkpoints and optimizer states untouched.'},
    'next_checks':{'joint_v9_not_before':j['time']+600,'v8_not_before':v8['time']+600,
        'v10_not_before':v10['time']+600,'v7_not_before':v7['time']+600},
    'next_actions':['Inspect joint v9 step20 validation and premature STOP trend. Analyze inherited repeat/collision versus false-STOP reward with controlled training-only comparisons before choosing a change.',
        'Compare paired calibration pilots after real validation and inspect state diversity, augmentation robustness and actual action dependence',
        'Investigate weak goal conditioning; consider automatically generated paired-goal expert data without manual predicates or policy maps.',
        'Monitor disk after lossless compression','Continue towards >=70% without manual predicates or a visual action bypass']}
path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'saved':str(path),'goal_achieved':False,'joint_update':j['status']['update'],'next_joint_check':j['time']+600}))
