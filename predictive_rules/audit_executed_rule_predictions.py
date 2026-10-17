"""Compare saved symbolic proofs to their subsequently observed visual labels.

Entirely post-execution, training-only. No observation is returned to a controller.
"""
import json,torch,numpy as np
from pathlib import Path
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config
from predictive_nesy.graph_subgoal_interpreter import Program
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');folder=root/'counterfactual_proof_v8r1'
manifest=json.loads((folder/'manifest.json').read_text());torch.set_num_threads(2)
saved=torch.load(manifest['policy_checkpoint']['path'],map_location='cpu',weights_only=False)
model=VisualMemorySubgoalModel(Config(**saved['model_config'])).eval();model.load_state_dict(saved['model']);rows=[]
with torch.no_grad():
    for file in sorted(folder.glob('*.npz'))[:48]:
        trace=json.loads((folder/'proofs'/file.stem/'trace.json').read_text());event=trace['programs'][0]
        assert event['start']==0
        pg=Program.load(folder/'proofs'/file.stem/event['file'])
        with np.load(file,allow_pickle=False) as data:
            n=min(len(event['plan']['actions']),len(data['actions']),10)
            codes=torch.from_numpy(np.concatenate([data['codes'][:n],data['actual'][n-1:n]],0).astype(np.int64))[None]
            task=torch.tensor([int(data['task'])]);acts=data['actions'][:n].tolist()
        generated=model.generate(task,codes[:,:1],torch.tensor([[17]]),torch.tensor([1]),start=0)
        generated['definitions']=torch.from_numpy(pg.definitions.copy())[None]
        grounded=model.encode_sequence(generated,codes)[0].argmax(-1)[0].tolist()
        expected=[s[0] for s in event['plan']['states'][:n+1]]
        initial_match=grounded[0]==expected[0]
        matching=[a==b for a,b in zip(grounded[1:],expected[1:])]
        rows.append(dict(episode=file.stem,initial_grounding_matches=initial_match,actions=acts,
            predicted=expected,observed=grounded,matches=matching,
            first_divergence=next((i+1 for i,m in enumerate(matching) if not m),None),
            observed_end_matches_generated_goal=grounded[-1]==int(pg.subgoal[0]),
            symbolic_end_matches_generated_goal=expected[-1]==int(pg.subgoal[0])))
usable=[r for r in rows if r['initial_grounding_matches']]
total=sum(len(r['matches']) for r in usable);right=sum(sum(r['matches']) for r in usable)
result=dict(training_only=True,post_execution_only=True,policy_checkpoint=manifest['policy_checkpoint'],
    samples=len(rows),initial_grounding_agreements=len(usable),predicted_states=total,correct_predicted_states=right,
    first_transition_correct=sum(r['matches'][0] for r in usable),
    entire_executed_prefix_correct=sum(all(r['matches']) for r in usable),
    observed_endpoint_equals_generated_goal=sum(r['observed_end_matches_generated_goal'] for r in usable),rows=rows)
(root/'diagnostics/executed_rule_predictions_v8r1.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
