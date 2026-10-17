"""Train-only counterfactual evidence for actions preferred by symbolic planning."""
from collections import defaultdict
import json,time
import numpy as np
import torch
from data import ROOT,sha
from model import FactoredWorld,Config,hard
from planner import Program,plan

@torch.inference_mode()
def main():
    torch.set_num_threads(3);began=time.time()
    run=ROOT/'run_grounded_memory_v42';commit=json.loads((run/'latest.COMMITTED.json').read_text())
    file=run/commit['filename'];assert sha(file)==commit['sha256']
    saved=torch.load(file,map_location='cpu',weights_only=False);model=FactoredWorld(Config(**saved['model_config'])).eval()
    model.load_state_dict(saved['model']);c=model.cfg
    groups=defaultdict(list)
    for p in sorted((ROOT/'onpolicy_successors_v40/counterfactual_onpolicy').glob('*.json')):
        record=json.loads(p.read_text());assert record['training_only'] and not record['goal_supervision']
        groups[(record['domain'],record['category'],record['distance_group'])].append((p,record))
    chosen=[pair for group in sorted(groups) for pair in groups[group][:3]];rows=[]
    for p,r in chosen:
        with np.load(p.with_suffix('.npz')) as data:
            x=torch.from_numpy(data['features'][:1].copy())[None]
            after=torch.from_numpy(data['successor_features'][0].copy())
        lo,z,_=model.observation_encoding(x);ob=hard(lo);previous=torch.tensor([[17]])
        lm,_,_=model.memory_sequence(ob,previous,z);m=int(lm[0,0].argmax());o=int(lo[0,0].argmax())
        task=torch.tensor([0 if r['category']=='chair' else 1]);context=model.context_sequence(x,ob,previous,task)[:,0]
        program=Program(**model.compile(context,task)).validate();proof=plan(program,m,o,32)
        source=m*c.observations+o;actual=model.observe(after).argmax(-1).numpy()
        pobs=program.outcome_probability[source];nextm=program.next_memory[source]
        same=np.mean((after.numpy().astype(np.float32)-x[0,0].numpy().astype(np.float32))**2,axis=(-1,-2))<1e-10
        correct=pobs.argmax(-1)==actual
        row=dict(episode=r['episode'],found=proof['found'],
            all_action_observation_accuracy=float(correct.mean()),
            all_action_visual_unchanged_rate=float(same.mean()),
            all_action_nonstop_visual_unchanged_rate=float(same[1:].mean()))
        if proof['found']:
            a=int(proof['actions'][0]);q=float(proof['model_goal_probability'])
            if a==0:
                continuation=float(nextm[a]*c.observations+actual[a]==program.goal)
            else:continuation=float(program._levels[31][nextm[a]*c.observations+actual[a]])
            row.update(action=a,selected_observation_correct=bool(correct[a]),
                selected_visual_unchanged=bool(same[a]),selected_actual_outcome_probability=float(pobs[a,actual[a]]),
                model_goal_probability=q,continuation_after_recorded_outcome=continuation,
                one_step_optimism=q-continuation,
                horizon4_probability=float(program._levels[4][source]),
                horizon8_probability=float(program._levels[8][source]),
                horizon16_probability=float(program._levels[16][source]))
        rows.append(row)
    selected=[r for r in rows if r['found']];moving=[r for r in selected if r['action']!=0]
    metrics={k:float(np.mean([r[k] for r in selected])) for k in selected[0] if k not in ('episode','action')}
    metrics.update(selected_nonstop_visual_unchanged_rate=float(np.mean([r['selected_visual_unchanged'] for r in moving])),
        selected_nonstop_observation_accuracy=float(np.mean([r['selected_observation_correct'] for r in moving])))
    result=dict(training_only=True,source_commit=commit,roots=len(rows),roots_with_plan=len(selected),
        metrics=metrics,rows=rows,seconds=time.time()-began,
        limitation='Single recorded action effects at training roots; continuation is still a model value, not physical success. No goal, reward, collision labels or map provided to inference.')
    (ROOT/'diagnostic_dinov2_v43/selected_effects.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
