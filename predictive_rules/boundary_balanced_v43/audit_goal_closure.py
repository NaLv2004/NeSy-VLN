"""Check learned goals and symbolic closure on existing successful training paths.

No simulator, no new validation and no runtime memory correction. Grounded
recorded symbols are available to scoring only; inference uses compiled rules.
"""
import json,time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from data import ROOT,sha
from model import FactoredWorld,Config,hard
from planner import Program,plan

@torch.inference_mode()
def main():
    torch.set_num_threads(3);began=time.time()
    old=ROOT/'run_grounded_memory_v42';commit=json.loads((old/'latest.COMMITTED.json').read_text())
    checkpoint=old/commit['filename'];assert sha(checkpoint)==commit['sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).eval();model.load_state_dict(saved['model']);c=model.cfg
    rows=json.loads((Path(saved['manifest']['data']['source'])/'training_rows.json').read_text())
    feature=np.load(ROOT/'prequant_visual_features_v18/features.npy',mmap_mode='r')
    probe=json.loads((ROOT/'diagnostic_dinov2_v43/dataset.json').read_text())
    episodes=defaultdict(list)
    for i,row in enumerate(rows):episodes[row['episode_id']].append(i)
    chosen=probe['heldout_episode_ids'][:36];report=[]
    for name in chosen:
        indices=episodes[name];n=len(indices)
        x=torch.from_numpy(np.concatenate([feature[indices],feature[indices[-1:]]],0))[None]
        actions=np.asarray([rows[i]['label'] for i in indices]);assert actions[-1]==0
        previous=torch.tensor([[17]+actions.tolist()]);task=torch.tensor([['chair','table'].index(rows[indices[0]]['category'])])
        logits,z,_=model.observation_encoding(x);observations=hard(logits)
        memory,_,_=model.memory_sequence(observations,previous,z)
        contexts=model.context_sequence(x,observations,previous,task)
        m=memory.argmax(-1)[0].numpy();o=logits.argmax(-1)[0].numpy()
        program=Program(**model.compile(contexts[:,0],task)).validate()
        expected_goal=int(m[-1])*c.observations+int(o[-1]);goal_m,goal_o=divmod(program.goal,c.observations)
        initial=int(m[0])*c.observations+int(o[0]);length=min(32,n)
        proof=plan(program,int(m[0]),int(o[0]),horizon=32)
        modal_m,modal_o=int(m[0]),int(o[0]);feedback_m=int(m[0])
        dist=np.zeros(c.memories*c.observations,np.float64);dist[initial]=1.
        known_m_correct=known_o_correct=0
        for t,a in enumerate(actions):
            source=int(m[t])*c.observations+int(o[t])
            known_m_correct+=int(program.next_memory[source,a]==m[t+1])
            known_o_correct+=int(program.outcome_probability[source,a].argmax()==o[t+1])
            source=modal_m*c.observations+modal_o
            modal_m=int(program.next_memory[source,a]);modal_o=int(program.outcome_probability[source,a].argmax())
            source=feedback_m*c.observations+int(o[t]);feedback_m=int(program.next_memory[source,a])
            target=np.zeros((c.memories,c.observations),np.float64)
            np.add.at(target,program.next_memory[:,a],dist[:,None]*program.outcome_probability[:,a])
            dist=target.ravel()
        report.append(dict(episode=name,actions=n,teacher_exceeds_action_budget=n>32,
            generated_goal_matches_recorded_endpoint=program.goal==expected_goal,
            generated_goal_memory_correct=goal_m==int(m[-1]),generated_goal_observation_correct=goal_o==int(o[-1]),
            generated_goal_equals_initial=program.goal==initial,
            goal_has_terminal_preimage=bool(np.any((program.next_memory[:,0]==goal_m)&(program.outcome_probability[:,0,goal_o]>0))),
            known_current_memory_accuracy=known_m_correct/n,known_current_observation_accuracy=known_o_correct/n,
            teacher_modal_endpoint_is_goal=modal_m*c.observations+modal_o==program.goal,
            teacher_distribution_goal_probability=float(dist[program.goal]),
            teacher_distribution_recorded_endpoint_probability=float(dist[expected_goal]),
            observation_feedback_endpoint_memory_correct=feedback_m==int(m[-1]),
            symbolic_plan_found=proof['found'],symbolic_plan_first_action=proof['actions'][0] if proof['found'] else None,
            first_teacher_action=int(actions[0]),first_action_matches_teacher=bool(proof['found'] and proof['actions'][0]==actions[0]),
            initial_predicted_goal_probability=proof['model_goal_probability']))
    metrics={key:sum(float(r[key]) for r in report)/len(report) for key in report[0] if key not in ('episode','symbolic_plan_first_action','first_teacher_action')}
    # Quantization aliasing diagnostic on the same inverse-probe observations.
    source_indices=probe['source_row_indices'];lookup={i:j for j,i in enumerate(source_indices)}
    codes=[]
    for begin in range(0,len(source_indices),256):
        codes.extend(model.observe(torch.from_numpy(feature[source_indices[begin:begin+256]].copy())).argmax(-1).tolist())
    fit_counts=defaultdict(lambda:np.zeros(17,np.int64));combined_counts=defaultdict(lambda:np.zeros(17,np.int64));test=[]
    for record in probe['records']:
        pair=(codes[lookup[record['current']]],codes[lookup[record['successor']]])
        combined_counts[pair][record['action']]+=1
        if record['split']=='fit':fit_counts[pair][record['action']]+=1
        else:test.append((pair,record['action']))
    majority=np.sum(list(fit_counts.values()),axis=0).argmax()
    hit=sum(int((fit_counts[p].argmax() if p in fit_counts else majority)==a) for p,a in test)
    compression=dict(codes=len(set(codes)),pairs=len(combined_counts),samples=len(probe['records']),
        empirical_all_sample_pair_majority_ceiling=sum(v.max() for v in combined_counts.values())/len(probe['records']),
        heldout_train_pair_lookup_accuracy=hit/len(test),
        note='Ceiling is empirical for this pooled diagnostic set, with no action-equivalence credit; lookup supplies no runtime actions')
    result=dict(training_only=True,source_commit=commit,episodes=len(report),metrics=metrics,compression=compression,
        rows=report,seconds=time.time()-began,
        limitation='Diagnostics on successful recorded training paths; not heldout navigation success. First-action mismatch can be a valid alternative route.')
    out=ROOT/'diagnostic_dinov2_v43/goal_closure.json';out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
