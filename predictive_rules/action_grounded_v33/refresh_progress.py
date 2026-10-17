import json
from pathlib import Path

here=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding='utf-8'))
record=read(here/'progress.json')
current=read(here/'monitor_latest.json')
status=current['status']
runtime=record['runtime']
runtime.update(process_alive_verified=bool(current['processes'].get(str(runtime['pid']))),
    command=current['processes'].get(str(runtime['pid'])),status=status.get('state'),
    latest_verified_update=status.get('update'),latest_commit=current['latest_commit'],
    observed_at_remote_epoch=current['observed_at'],
    next_routine_check_not_before_remote_epoch=current['observed_at']+600,
    last_update_seconds=status.get('seconds'),memory_peak_gib=status.get('memory_peak_gib'))
record.update(scheduled_development_results=current['validation'],training_window_metrics=current['training_windows'],
    previous_goal_turn_classification='progress: v33 implemented and actual updates verified',
    this_turn_classification='progress: implemented and uploaded hard multi-step dynamics supervision as isolated v34 source, not launched; v33 independently verified live at1463 with existing automatic evaluations20/80 and15/80',
    next_prepared_revision=read(here.parent/'open_loop_v34/prepared.json'),
    conclusion='Existing scheduled v33 results: step50020/80,step100015/80. Both frequency>=5. Peak is slightly higher than prior19/80, but stable improvement and the60% goal are unproven.',
    next_action='Keep current v33 running; only two scheduled evaluations completed. No extra evaluations. No routine read before recorded600s gate. Preserve best step500. Hard open-loop v34 source is prepared remotely, but has no process or launch. Do not assume v34 is running. Reconsider it after the existing formal run has additional scheduled evidence or terminates.')
(here/'progress.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(update=status.get('update'),successes=[x['successes'] for x in current['validation']],
    next_check=current['observed_at']+600,prepared_revision_running=False)))
