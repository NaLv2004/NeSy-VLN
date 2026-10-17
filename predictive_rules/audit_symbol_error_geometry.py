"""Post-execution latent error sizes using saved rules, never policy feedback."""
import json
from pathlib import Path
import numpy as np
import torch
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config
from predictive_nesy.graph_subgoal_interpreter import Program
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');torch.set_num_threads(2)
folder=root/'counterfactual_proof_v11';checkpoint=root/'run_vocabulary_rules_v11/checkpoint_003200.pt'
saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
m=VisualMemorySubgoalModel(Config(**saved['model_config'])).eval();m.load_state_dict(saved['model']);rows=[]
with torch.no_grad():
    for file in sorted(folder.glob('*.npz')):
        trace=json.loads((folder/'proofs'/file.stem/'trace.json').read_text());event=trace['programs'][0]
        executable=Program.load(folder/'proofs'/file.stem/event['file'])
        with np.load(file,allow_pickle=False) as data:
            n=min(10,len(event['plan']['actions']),len(data['actions']))
            codes=torch.tensor(np.concatenate([data['codes'][:n],data['actual'][n-1:n]],0).astype(np.int64))[None]
            task=torch.tensor([int(data['task'])])
        p=m.generate(task,codes[:,:1],torch.tensor([[17]]),torch.tensor([1]),start=0)
        p['definitions']=torch.tensor(executable.definitions.copy())[None]
        posterior=m.encode_sequence(p,codes)[0][0].numpy();labels=posterior.argmax(-1)
        assert labels[0]==event['initial_state'][0]
        for t,step in enumerate(trace['steps'][:n]):
            after=int(step['after'][0]);before=int(step['before'][0]);action=int(step['action']);actual=int(labels[t+1])
            scores=posterior[t+1];rank=int((scores>scores[after]).sum())+1
            row=executable.transitions[before,action]
            rows.append(dict(episode=file.stem,offset=t+1,action=action,predicted=after,actual=actual,
                correct=after==actual,rank_of_predicted_symbol_in_observed_posterior=rank,
                cosine_shortfall_from_best=float(np.log(scores.max()/max(scores[after],1e-30))/8),
                learned_rule_confidence=float(row.max()),probability_of_actual_successor=float(row[actual]),
                inferred_symbol_appears_in_actual_window=bool(after in labels)))
def summary(selected):
    return dict(steps=len(selected),exact=sum(r['correct'] for r in selected),
        top5=sum(r['rank_of_predicted_symbol_in_observed_posterior']<=5 for r in selected),
        top10=sum(r['rank_of_predicted_symbol_in_observed_posterior']<=10 for r in selected),
        median_rank=float(np.median([r['rank_of_predicted_symbol_in_observed_posterior'] for r in selected])),
        mean_cosine_shortfall=float(np.mean([r['cosine_shortfall_from_best'] for r in selected])),
        mean_rule_confidence=float(np.mean([r['learned_rule_confidence'] for r in selected])),
        mean_probability_actual=float(np.mean([r['probability_of_actual_successor'] for r in selected])),
        inferred_symbol_never_in_actual_window=sum(not r['inferred_symbol_appears_in_actual_window'] for r in selected))
result=dict(training_only=True,post_execution=True,no_new_navigation=True,saved_runtime_programs_used=True,
    all=summary(rows),movement=summary([r for r in rows if r['action']]),stop=summary([r for r in rows if not r['action']]),rows=rows)
(root/'diagnostics/v11_symbol_error_geometry.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
