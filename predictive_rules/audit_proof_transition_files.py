"""Audit a fixed prefix of committed collection files, not job progress."""
import argparse,json
from pathlib import Path
import numpy as np
from predictive_nesy.graph_subgoal_interpreter import Program,successors
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
parser=argparse.ArgumentParser();parser.add_argument('--folder',default=str(root/'counterfactual_proof_v8r1'))
parser.add_argument('--out',default=str(root/'diagnostics/proof_transition_file_audit.json'))
parser.add_argument('--limit',type=int,default=16);args=parser.parse_args();folder=Path(args.folder)
manifest=json.loads((folder/'manifest.json').read_text());allowed=set(manifest['episodes'])
held_out={f'layout_{i}' for i in [116,117,122,123,*range(200,208)]}
assert not held_out & set(manifest['data']['train_layouts']) and manifest['training_only']
records=[]
files=sorted(folder.glob('*.npz'));selected=files[:args.limit] if args.limit else files
for file in selected:
    assert file.stem in allowed
    with np.load(file,allow_pickle=False) as d:
        assert set(d.files)=={'codes','successors','actual','actions','probe_valid','probe_rgb_error','task'}
        n=len(d['actions']);assert d['codes'].shape==(n,49) and d['successors'].shape==(n,17,49)
        assert ((d['actions']>=0)&(d['actions']<17)).all()
        assert np.array_equal(d['codes'][1:],d['actual'][:-1])
        assert ((d['probe_rgb_error']<.5)==d['probe_valid']).all()
        actions=d['actions'].tolist();valid=int(d['probe_valid'].sum())
    trace=json.loads((folder/'proofs'/file.stem/'trace.json').read_text())
    assert trace['episode']['scene'] in manifest['data']['train_layouts'] and trace['episode']['scene'] not in held_out
    assert trace['neural_calls_outside_boundaries']==0 and actions==[s['action'] for s in trace['steps']]
    proofs={}
    for e in trace['programs']:
        program=Program.load(folder/'proofs'/file.stem/e['file']);plan=e['plan'];state=np.array(e['initial_state'])
        assert program.digest()==plan['program_sha256'] and e['start']%10==0
        assert e['observed_frame_indices']==list(range(max(0,e['start']-10),e['start']+1))
        if not plan['found']:assert not plan['actions'];continue
        assert plan['actions'] and 0 not in plan['actions'][:-1]
        assert len(plan['actions'])==min(10,32-e['start']) or plan['actions'][-1]==0
        assert len(plan['states'])==len(plan['actions'])+1
        for i,action in enumerate(plan['actions']):
            assert state.tolist()==plan['states'][i]
            before=state.tolist();state=successors(program,state)[0][action]
            assert state.tolist()==plan['states'][i+1]
            proofs[e['start']+i]=(action,before,state.tolist(),program.digest())
        assert np.array_equal(state,program.subgoal)
    for step in trace['steps']:
        assert proofs[step['step']]==(step['action'],step['before'],step['after'],step['program_sha256'])
    records.append(dict(episode=file.stem,states=n,valid_states=valid))
assert records
skips=list((folder/'proofs').glob('*/NO_PROOF.json'))
if args.limit==0:
    assert {p.stem for p in files}.isdisjoint({p.parent.name for p in skips})
    assert {p.stem for p in files}|{p.parent.name for p in skips}==allowed
    for path in skips:
        r=json.loads((path.parent/'trace.json').read_text())
        assert not r['steps'] and r['neural_calls_outside_boundaries']==0
        assert r['episode']['scene'] in manifest['data']['train_layouts']
        assert not r['programs'][0]['plan']['found'] and not r['programs'][0]['plan']['actions']
result=dict(files_audited=len(records),training_only=True,held_out_layouts_absent=True,visual_and_action_fields_only=True,
    all_executed_actions_have_complete_exact_proofs=True,observations_do_not_correct_policy_between_revisions=True,
    total_states=sum(r['states'] for r in records),valid_states=sum(r['valid_states'] for r in records),records=records,
    full_collection_audited=args.limit==0,no_proof_skips_audited=len(skips) if args.limit==0 else 0)
Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}))
