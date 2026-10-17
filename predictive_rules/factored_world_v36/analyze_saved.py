"""CPU-only inspection of completed evaluation proofs and cached train data."""
from collections import Counter
import json
from pathlib import Path
import numpy as np

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'run_factored_world_v36'
summaries={}
for update in [500,1000,1500,2000]:
    folder=OUT/f'dev_{update:06d}'
    traces=list(folder.glob('*/trace.json'))
    if len(traces)!=80:continue
    counts=Counter();stop_confidence=[];stop_goal_density=[];examples=[]
    for path in traces:
        trace=json.loads(path.read_text());success=bool(trace['metrics'].get('success',False))
        counts['episodes']+=1;counts['successes']+=success
        for event in trace['programs']:
            with np.load(path.parent/event['file']) as p:
                table=p['next_state'];goal=int(p['goal'])
                stop_goal_density.append(float((table[:,0]==goal).mean()))
                counts['compiled_stop_rows']+=len(table)
                counts['compiled_stop_changes_observation']+=int((table[:,0]%64!=np.arange(len(table))%64).sum())
        if trace['steps'] and trace['steps'][-1]['action']==0:
            step=trace['steps'][-1];goal=step['proof']['goal'];counts['stops']+=1
            counts['false_stops']+=not success
            counts['stops_at_first_action']+=step['step']==0
            counts['stop_observation_equals_goal_observation']+=step['observed']==goal%64
            counts['stop_predicts_changed_observation']+=step['observed']!=step['predicted_observation']
            counts['false_stop_with_changed_observation']+=(not success and step['observed']!=step['predicted_observation'])
            event=next(p for p in trace['programs'] if p['digest']==step['proof']['program_sha256'])
            with np.load(path.parent/event['file']) as p:
                stop_confidence.append(float(p['confidence'][step['proof']['states'][0],0]))
            if not success and len(examples)<5:
                examples.append(dict(episode=trace['episode']['episode_id'],step=step['step'],observed=step['observed'],
                    predicted_observation=step['predicted_observation'],goal=goal,metrics=trace['metrics']))
    summaries[str(update)]=dict(counts=counts,mean_stop_confidence=float(np.mean(stop_confidence)),
        mean_fraction_table_nodes_with_stop_to_generated_goal=float(np.mean(stop_goal_density)),false_stop_examples=examples)
train=[]
files=sorted((ROOT/'raw_successor_features_v31').glob('counterfactual_*/*.npz'))
for path in files[::max(1,len(files)//48)]:
    with np.load(path) as data:
        valid=data['probe_valid'];current=data['features'][valid].astype(np.float32);successors=data['successor_features'][valid].astype(np.float32)
        if not len(current):continue
        train.append(dict(file=str(path),positions=len(current),
            stop_mse=float(((current-successors[:,0])**2).mean()),
            stop_max_abs=float(np.max(np.abs(current-successors[:,0]))),
            stop_token_equality=float((data['recorded_codes'][valid]==data['recorded_successors'][valid,0]).all(-1).mean()),
            mean_action_mse=((current[:,None]-successors)**2).mean((0,2,3)).tolist()))
result=dict(completed_development_proofs=summaries,training_cache_stop_identity=dict(files=len(train),
    positions=sum(t['positions'] for t in train),max_abs=max(t['stop_max_abs'] for t in train),
    mean_mse=float(np.mean([t['stop_mse'] for t in train])),
    all_stop_tokens_equal=all(t['stop_token_equality']==1 for t in train)),
    scope='Only already completed dev traces and existing train feature cache; no live trainer polling, model inference or new rollouts')
(OUT/'saved_proof_analysis.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
