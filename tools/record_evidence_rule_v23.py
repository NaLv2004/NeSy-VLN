import json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules';e=root/'evidence'
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
p=read(root/'goal_progress.json');launcher=read(e/'evidence_rule_v23_launcher.json');pre=read(e/'preflight_evidence_rule_v23.json')
assert pre['passed'] and pre['between_revision_neural_call_rejected']
audit=read(e/'closed_loop_states_v22.json');ident=read(e/'transition_identifiability_v22.json')
p['closed_loop_states_v22']={k:v for k,v in audit.items() if k!='episodes'}
p['transition_identifiability_v22']={k:v for k,v in ident.items() if k not in ['initial_observed_branches','conflicts']}
p['active_runs']['evidence_rule_v23']=dict(launcher=launcher,preflight=pre,last_observed_state='launched',last_observed_update=None,
    hypothesis='Global context compression may impair generated rule rows; source-conditioned attention exposes current boundary visual patches without per-action inference.',
    no_performance_improvement_established=True)
p.update(active_scope='Train complete source-evidence-conditioned rules and an independent discrete subgoal; verify unchanged exact symbolic execution',
    last_verified_pid=launcher['pid'],last_verified_update=None,last_remote_check_unix=launcher['time'],next_regular_check_not_before_unix=launcher['time']+660,
    goal_status='active',completion_proven=False)
p['pending']=[
    'V23 launched from immutable v21step1000 on GPUs3/4/5; GPU7 evaluates development80 at500/1000. Source src_evidence_rule_v23, run run_evidence_rule_v23. Never duplicate on a timeout; verify this PID and source first.',
    'One active kernel process-exit wait is attached through local unified-exec session3054. It saves diagnostics/evidence_rule_v23_completed.json on actual completion; no periodic job/GPU polling occurs in that wait.',
    'V22 completed offline audits. On32 saved training-task traces, initial grounding matches43/43 programs; autonomous47/216 and one-step73/216. First divergence is at action1 in23/36 executed programs. Failed tasks attain generated observed goal0/27; successful7/9.',
    'Across all137122 recorded action branches, current v21 dynamics47.594%; deterministic label-fit ceiling89.643% due contradictory labels in the same generated-program/state/action row. Both fitting and state aliasing remain issues; no single cause is proven.',
    'V23 attention only changes neural rule generation at boundaries. All37 original grounding/definition tensors stay frozen. Same data, losses, sampling, goal head initialization, exact solver and train/dev/final partitions. No manual semantic states, value/reward tables, maps, direct action fallback or inside-block correction.',
    'Preflight: initial programs bitwise identical to v21; future padded inputs cannot change program after a real update; between-revision neural call rejected. Goal gradients to rule branch0, world gradients to goal0, new attention gradient nonzero; real512+128 update0.259s, peak11.135GiB.',
    'Preserve unrelated GPU0/1/2/6 processes. All existing final validation tasks remain held out. Best completed exact-symbolic dev remainsv12step1600=5/80, reference43/80dev and44/80fixed; full execution-speed parity also unproven.'
]
p['previous_turn_classification']='progress: completed saved-trajectory audit and full transition identifiability audit; implemented a causal source-evidence generator and launched a bounded verified experiment; goal still unmet'
(root/'goal_progress.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(goal_complete=False,active_run='run_evidence_rule_v23',pid=launcher['pid'])))
