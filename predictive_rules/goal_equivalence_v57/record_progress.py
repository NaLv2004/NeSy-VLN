"""Record completed predecessor evidence and current acknowledged run."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
v56=BASE/'definition_rules_v56';handoff=read(v56/'handoff_result.json')
assert handoff['state']=='training_dispatched'
assert handoff['predecessor_terminal']['state']=='finished' and handoff['predecessor_terminal']['exit_code']==0
oldpath=BASE/'temporal_symbols_v55/experiment_record.json';old=read(oldpath)
old.update(phase='finished',terminal=handoff['predecessor_terminal'],development=handoff['predecessor_development'],
    terminal_successes=11,best_successes=11,completion_proven=False,
    conclusion='Temporal density-ratio training did not improve the v52 source best11/80; final observation prediction21.69%.')
write(oldpath,old)
recordpath=v56/'experiment_record.json';record=read(recordpath)
record.update(phase='training',supervisor_pid=handoff['supervisor_pid'],handoff_result=handoff,
    startup_evidence=dict(source='User-requested monitor service restart and live SSH telemetry',
        local_received_at=1792196791.0931706,connected=True,stale=False,alive=True,update=456),
    goal_grounding_audit=read(v56/'goal_grounding.json'),
    next_action='Read complete unchanged dev80 results at the next permitted process check. Do not infer navigation from CPU pilot metrics.')
write(recordpath,record)
gpath=BASE/'goal_progress.json';goal=read(gpath)
goal.update(authoritative_record=str(recordpath),remote_run='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/run_definition_rules_v56',
    remote_source='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/src_definition_rules_v56',
    phase='training',supervisor_pid=handoff['supervisor_pid'],rank_pids=[2155569],
    rank_pid_coverage='rank0 from telemetry; rank1 awaits next full process inspection',latest_actual_update=456,
    last_user_requested_dashboard_check_local_unix=1792196791.0931706,
    next_regular_check_not_before_unix=1792197421.0931706,
    next_action=record['next_action'],previous_goal_turn_classification='progress',
    current_goal_turn_classification='progress',
    independent_experiment=str(Path(__file__).parent/'experiment_record.json'))
goal.pop('pending_successor',None);write(gpath,goal)
probe=dict(goal_status='active',completion_proven=False,phase='CPU_probe_running',
    objective='Improve learned goal-relative symbols without any neural planning bypass',
    remote_output='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015/diagnostic_goal_equivalence_v57',
    observed_issue='Across successful training demonstrations of the same task, endpoint O agreement under one fixed program was28.1%, despite93.75% own endpoint joint goal prediction.',
    hypothesis='The exact generated goal may be tied to one demonstrated endpoint appearance rather than a transferable success representation.',
    change='Add only a train-time negative log categorical agreement between two successful terminal images of the same requested category. Retain all v56 transition, inverse, reconstruction, usage and goal objectives.',
    no_manual_symbol_assignment=True,no_nonterminal_failure_labels=True,
    original_dev80_used=False,finalfixed80_used=False,live_training_modified=False,
    caveat='Same task success alone does not imply identical effects for all non-STOP actions. Check prediction and code diversity before considering promotion.',
    reference=dict(title='Bisimulation Makes Analogies in Goal-Conditioned Reinforcement Learning',
        url='https://proceedings.mlr.press/v162/hansen-estruch22a.html',
        inspected='Definitions3.1/4.1 and equations4-6',
        distinction='Research motivates functional goal-conditioned representations and preserving dynamics. This pilot is a simpler task-supervised endpoint agreement auxiliary, NOT a reproduction or proof of GCB.'),
    source_sha256=hashlib.sha256((Path(__file__).parent/'probe_goal_equivalence.py').read_bytes()).hexdigest())
write(Path(__file__).parent/'experiment_record.json',probe)
print(json.dumps(dict(current_run=goal['remote_run'],v55_terminal_successes=11,v56_acknowledged_update=456,next_check=goal['next_regular_check_not_before_unix'])))
