"""Record the authoritative run and retire only this project's previous status."""
import json
from pathlib import Path
here=Path(__file__).resolve().parent
snapshot=json.loads((here/'startup.json').read_text(encoding='utf-8'))
progress=dict(state='running_verified_ppo_updates',remote_root='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015',
    source_directory='src_program_ppo_v35',run_directory='run_program_ppo_v35',gpus=[6,7],
    supervisor=snapshot['supervisor'],launcher=snapshot['launch'],last_observed_at=snapshot['observed_at'],
    next_remote_check_not_before=snapshot['observed_at']+600,
    latest_commit=snapshot['latest'],last_update=snapshot['updates'][-1],
    current_gpu_snapshot=snapshot['gpu'],goal_complete=False,
    constraints=['Originaldevelopment80 unchanged; fixed80 untouched',
        'All executed actions derive from backward hard-goal proofs',
        'PPO trains rule/goal context encoders; downstream decoders, anonymous perception and renewalGRU fixed',
        'No supervised action or dynamics auxiliary loss',
        'Every10updates scheduleddevelopment; no extra diagnostic evaluations',
        'Preserve all prior v31/v32/v33 checkpoints; rolling new latest and previous plus best',
        'Do not touch unrelated GPU0–5 jobs'],
    numerical_fix='Disable TF32 in both sampled program density and replay; strict replay check passes',
    prepared_but_not_launched='open_loop_v34 remains unlaunched; PPO takes priority')
(here/'progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
old=here.parent/'action_grounded_v33/progress.json'
if old.exists():
    value=json.loads(old.read_text(encoding='utf-8'))
    value.update(state='superseded_by_user_authorized_program_ppo_v35',replacement=str(here/'progress.json'),all_saved_checkpoints_preserved=True)
    old.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(update=progress['last_update']['update'],next_check=progress['next_remote_check_not_before'])))
