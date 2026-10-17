"""Read existing training probes to check physical and categorical integrity."""
import json,time
from pathlib import Path
import numpy as np
from data import ROOT,sha
from evaluate import atomic
DATA=ROOT/'counterfactual_closure_v63'
manifest=json.loads((DATA/'manifest.json').read_text());samples=[]
for p in sorted((DATA/'episodes').glob('*.json'))[:8]:
    r=json.loads(p.read_text());assert r['training_only'] and r['offline_future_labels_never_assigned_to_policy']
    assert r['scene'] in manifest['train_layouts'] and r['episode'] in manifest['episode_ids']
    assert r['source_sha256']==manifest['source_commit']['sha256'] and sha(p.with_suffix('.npz'))==r['sha256']
    with np.load(p.with_suffix('.npz'),allow_pickle=False) as a:
        n=len(a['memory']);assert n==r['positions']
        assert a['context'].shape==(n,128) and np.isfinite(a['context']).all()
        for key in ('observation','first_observation','second_observation'):assert a[key].min()>=0 and a[key].max()<64
        for key in ('memory','first_memory','second_memory'):assert a[key].min()>=0 and a[key].max()<32
        assert a['second_actions'].shape==(n,17,2) and (a['second_actions'][:,:,0]!=a['second_actions'][:,:,1]).all()
        assert not a['second_valid'][:,0].any() and a['second_valid'][:,1:].all()
        for j,state in enumerate(r['steps']):
            assert a['memory'][j]==state['memory'] and a['observation'][j]==state['observed']
            assert a['recorded_actions'][j]==state['proof']['actions'][0]==state['action']
        times=[x['start'] for x in r['programs']];assert all(b-a>=5 for a,b in zip(times,times[1:]))
    assert all(e<.5 for e in r['two_step_probe_execution_errors'])
    samples.append(dict(episode=r['episode'],positions=n,executed_actions=r['executed_actions'],
        two_step_execution_errors=r['two_step_probe_execution_errors'],termination=r['termination']))
assert samples,'No finished samples yet'
record=dict(time=time.time(),samples=samples,training_only=True,source_sha256=manifest['source_commit']['sha256'],
    masked_terminal_parent=True,recorded_integer_premises_match_executed_proofs=True,
    original_complete_validation_sets_untouched=True,scope='Existing first8 sorted completed episodes; not a navigation-performance evaluation')
atomic(DATA/'sample_contract.json',record);print(json.dumps(record))
