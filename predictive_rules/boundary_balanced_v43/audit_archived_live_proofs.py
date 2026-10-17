"""Replay already-recorded symbolic proofs; no environment or new rollout."""
import json,os,time
from pathlib import Path
import torch
from program_archive import restore
from planner import audit
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.15)
folder=ROOT/'run_boundary_balanced_v43/dev_005000'
assert (folder/'summary.json').exists()
checked={}
for path in sorted(folder.glob('*/trace.json')):
    trace=json.loads(path.read_text())
    for entry in trace['programs']:
        rank_file=entry['model_archive']['file']
        if rank_file in checked:continue
        steps=[s for s in trace['steps'] if s['step']==entry['start']]
        if not steps:continue
        program=restore(path.parent/entry['file'])
        assert program.digest()==entry['digest']
        audit(program,steps[0]['proof'])
        checked[rank_file]=dict(episode=trace['episode']['episode_id'],step=entry['start'],
            table_sha256=program.digest(),first_action=steps[0]['action'],proof_replay_exact=True)
    if len(checked)==2:break
assert len(checked)==2
report=dict(time=time.time(),evaluation_update=5000,checks=checked,new_environment_rollouts=0)
(ROOT/'diagnostic_program_archive_v43/live_proof_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
