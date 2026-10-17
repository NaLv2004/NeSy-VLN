"""Independently check saved navigation state-transition proofs without torch.

Does not call the planner/controller implementation. It directly indexes the
exported rule arrays and compares every recorded successor and executed plan.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(folder):
    folder=Path(folder)
    trace=json.loads((folder/'trace.json').read_text())
    programs={item['start']:item for item in trace['programs']}
    if not programs:
        raise ValueError('This is not an exported predictive-rule trajectory')
    checked=0
    for start,record in programs.items():
        assert start%10==0
        assert record['observed_frame_indices']==list(range(max(0,start-10),start+1))
        with np.load(folder/record['file'],allow_pickle=False) as file:
            arrays={key:file[key] for key in file.files}
        h=hashlib.sha256()
        fields=['parents','transitions','rewards','values','weights','definitions'] if 'rewards' in arrays else ['parents','transitions','risks','goals','values','weights','definitions']
        if 'rewards' in arrays:
            assert set(arrays)==set(fields)|{'start','interval'}
        for name in fields:
            h.update(np.ascontiguousarray(arrays[name]).tobytes())
        h.update(str((int(arrays['start']),int(arrays['interval']))).encode())
        digest=h.hexdigest()
        assert digest==record['plan']['program_sha256']
        state=np.asarray(record['initial_state'],dtype=np.int64)
        belief=np.eye(arrays['transitions'].shape[-1],dtype=np.float32)[state]
        selected=[s for s in trace['steps'] if s['program_start']==start]
        assert [s['action'] for s in selected]==record['plan']['actions'][:len(selected)]
        for index,step in enumerate(selected):
            assert step['step']==start+index
            assert step['program_sha256']==digest
            np.testing.assert_array_equal(state,step['before'])
            if 'before_belief' in step:
                np.testing.assert_allclose(belief,step['before_belief'],atol=2e-5,rtol=2e-5)
                if step['action']:
                    next_belief=[]
                    for factor,(left,right) in enumerate(arrays['parents']):
                        if left==right:
                            premises=np.diag(belief[left])
                        else:
                            premises=np.outer(belief[left],belief[right])
                        probability=np.sum(premises[...,None]*arrays['transitions'][factor,step['action']],axis=(0,1))
                        next_belief.append(probability)
                    belief=np.asarray(next_belief)
                np.testing.assert_allclose(belief,step['after_belief'],atol=2e-5,rtol=2e-5)
                state=np.asarray(step['after_belief']).argmax(-1)
                np.testing.assert_array_equal(state,step['after'])
                checked+=1
                continue
            if step['action']!=0:
                parent=arrays['parents'];f=np.arange(len(state))
                probabilities=arrays['transitions'][f,step['action'],state[parent[:,0]],state[parent[:,1]]]
                state=probabilities.argmax(-1)
            else:
                probabilities=np.eye(arrays['transitions'].shape[-1])[state]
            np.testing.assert_array_equal(state,step['after'])
            np.testing.assert_allclose(probabilities,step['transition_probabilities'],atol=0,rtol=0)
            checked+=1
    assert checked==len(trace['steps'])
    assert trace['neural_calls_outside_boundaries']==0
    return dict(episode_id=trace['episode']['episode_id'],steps=checked,programs=len(programs),
                all_state_transitions_exact=True,all_actions_from_saved_plan=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('evaluation')
    args=parser.parse_args()
    directory=Path(args.evaluation)
    records=[audit(path.parent) for path in sorted(directory.glob('*/trace.json'))]
    report=dict(episodes=len(records),steps=sum(r['steps'] for r in records),
        programs=sum(r['programs'] for r in records),all_verified=True,records=records)
    if not records:raise ValueError('No evaluation trajectories to audit')
    (directory/'independent_rule_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key!='records'},indent=2))


if __name__=='__main__':main()
