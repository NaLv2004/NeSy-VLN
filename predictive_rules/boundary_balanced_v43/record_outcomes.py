"""Local research ledger; does not change or restart training."""
import json,time
from pathlib import Path
here=Path(__file__).resolve().parent
record=json.loads((here/'experiment_record.json').read_text())
record['latest_verified_execution']=dict(remote_epoch=1792186047.2146614,update=4000,state='evaluating',
    supervisor_pid=2050991,torchrun_pid=2050992,rank0_pid=2050997,
    checkpoint_sha256='9a991ac1b265fb73cfe248695fa878f4fdb1874c8816cc577c72d0081b82fcb4',
    training_update_seconds=.31213808059692383)
record['completed_development']=json.loads((here/'development_003500.json').read_text())
record['effect_on_navigation']='First completed v43 dev80 evaluation is 11/80 versus v42 final 4/80. A single evaluation is not proof of a stable improvement. Target >80% unmet; finalfixed80 untouched.'
record['next_normal_status_poll_not_before_local_epoch']=1792186740.
record['goal_closure_diagnostic']=json.loads((here/'goal_closure.json').read_text())
record['selected_effects_diagnostic']=json.loads((here/'selected_effects.json').read_text())
record['diagnostic_conclusions']=[
    'Known successful training endpoints are usually represented by generated goals (32/36), but only19/36 modal teacher rollouts reach that generated goal.',
    'Planned first actions have38.9% observation prediction accuracy versus52.9% across all recorded17 action branches on36 training roots.',
    'Single-step model continuation optimism was only0.00074 on average; this does NOT establish that the first action alone explains failure.',
    'Do not compare teacher-path goal probability with optimal32-step probability as a controlled causal result: their horizons differ.',
    'The observed64-code pair classification retains action information on this corpus; this does not prove the code is Markov or useful for new scenes.',
    'DINOv2 direct replacement did not help this small inverse probe. Existing ConvNeXt was already trained on the broader corpus; the holdout applies to the probe head, not the encoder.'
]
record['next_research_candidate']=dict(status='researched_not_implemented',
    reference='https://github.com/deep-spin/entmax',paper='https://www.jmlr.org/papers/v21/19-021.html',
    motivation='Softmax world rows assign nonzero support to every future observation. Examine learned sparse transition support with proper Fenchel-Young losses, rather than arbitrary probability clipping.',
    requirements=['Keep all executed actions from symbolic derivation','Learn support from training outcomes; no handmade semantic states','A zero-probability true target must still have a corrective gradient','Do not deploy before fixed-training checks; current v43 remains running'])
(here/'experiment_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
probe=json.loads((here/'visual_probe.json').read_text())
probe['limitations']+=' Episode holdout applies only to this probe head and DINO PCA. The existing ConvNeXt encoder was previously trained on the broader corpus; this is a drop-in diagnostic, not unseen-domain pretraining comparison.'
(here/'visual_probe.json').write_text(json.dumps(probe,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(recorded=True,goal_achieved=False,development_successes=11,development_episodes=80,
    next_status_poll_not_before=record['next_normal_status_poll_not_before_local_epoch'])))
