"""Independently replay committed navigation actions from integer symbols only."""
import hashlib,json,time
from pathlib import Path
import numpy as np
base=Path('/home/user/xjy/nesy_vla_20261010/spatial_automaton_v43')
read=lambda p:json.loads(p.read_text())
commit=read(base/'COMMITTED');blob=(base/'programs.json').read_bytes()
assert hashlib.sha256(blob).hexdigest()==commit['program_sha256']
programs=json.loads(blob)

def infer(task,state):
    assert len(state)==99 and all(type(v)is int for v in state)
    assert all(0<=v<64 for v in state[:49]) and all(0<=v<=64 for v in state[49:98])
    assert state[98] in list(range(17))+[64]
    weights=np.zeros(17)
    for tree in programs[task]:
        node=0
        while tree['predicate'][node]>=0:
            slot,token=divmod(tree['predicate'][node],65)
            node=tree['true_branch' if state[slot]==token else 'false_branch'][node]
        conclusion=np.asarray(tree['conclusion'][node])
        assert (conclusion>=0).all() and abs(conclusion.sum()-1)<1e-6
        weights+=conclusion
    return weights/len(programs[task])

rows=[];errors=[];changes=0;total=0
for marker in sorted((base/'development').glob('*/evaluation.json')):
    episode=read(marker);trace=read(marker.parent/'trace.json');task=episode['category']
    assert len(trace['steps'])==episode['steps'] and 1<=len(trace['steps'])<=32
    for step in trace['steps']:
        p=infer(task,step['symbols']);error=float(np.max(np.abs(p-np.asarray(step['probabilities']))))
        errors.append(error);assert error<1e-10 and int(p.argmax())==step['action']
        for proof in step['proof']:
            assert all((step['symbols'][slot]==token)==truth for slot,token,truth in proof['clauses'])
        alternate=list(step['symbols']);alternate[:49]=[(v+1)%64 for v in alternate[:49]]
        changes+=int(int(infer(task,alternate).argmax())!=step['action']);total+=1
    rows.append(episode['episode_id'])
record={'time':time.time(),'episodes':len(rows),'actions':total,'probability_max_error':max(errors,default=0),
        'counterfactual_current_token_changes':changes,'program_sha256':commit['program_sha256'],
        'only_integer_symbols_loaded':True,'image_encoder_loaded':False,
        'qualification':'Cyclic token substitution measures dependence on symbolic input, not physical plausibility or navigation improvement.',
        'episode_ids':rows}
(base/'proof_audit.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
