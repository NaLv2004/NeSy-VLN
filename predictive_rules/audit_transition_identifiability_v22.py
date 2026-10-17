"""Measure deterministic rule identifiability using recorded training outcomes.

No new simulator trajectory, optimizer update, evaluation task, or runtime
observation correction is involved. Actual-source labels are audit targets.
"""
import hashlib,json,os,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from predictive_nesy.data import sha
from predictive_nesy.separated_goal_world import SeparatedGoalWorld,Config
from predictive_nesy.replayed_boundary_data import ReplayedBoundaryFeatures
from predictive_nesy.subgoal_data import TransitionBank

@torch.no_grad()
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='7'
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
    output=root/'diagnostics/transition_identifiability_v22.json';assert not output.exists()
    path=root/'run_separated_goal_v21/checkpoint_001000.pt'
    assert sha(path)==json.loads(path.with_suffix('.COMMITTED.json').read_text())['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=SeparatedGoalWorld(Config(**saved['model_config'])).cuda().eval();model.load_state_dict(saved['model'])
    features=ReplayedBoundaryFeatures(root/'prequant_visual_features_v18',root/'replayed_boundary_features_v20','cuda')
    rules=defaultdict(Counter);programs=defaultdict(Counter);stats={};began=time.time()
    matched_steps=[];sources=[TransitionBank(x['path'],saved['manifest']['data']['train_layouts']) for x in saved['manifest']['transition_banks']]
    for bank in sources:
        counts=Counter();confidence=defaultdict(list)
        for (start,offset),records in sorted(bank.by_key.items()):
            for begin in range(0,len(records),96):
                selected=records[begin:begin+96];rows=[];keys=[]
                for name,t in selected:
                    d=bank.episodes[name];first=max(0,start-10)
                    previous=np.array([int(d['actions'][j-1]) if j else 17 for j in range(first,start+1)])
                    history=d['codes'][first:start+1]
                    row=dict(task=int(d['task']),history=history,previous_actions=previous,
                        prefix_codes=d['codes'][start:t+1],successors=d['successors'][t])
                    rows.append(row)
                    digest=hashlib.sha256(history.tobytes()+previous.tobytes()).hexdigest()
                    # Same episode, causal visual/action prefix, task and revision
                    # means the same generated immutable program across banks.
                    keys.append((name,start,int(d['task']),digest))
                batch={k:torch.tensor(np.asarray([r[k] for r in rows]),device='cuda').long() for k in rows[0]}
                batch.update(start=start,episode_ids=[name for name,t in selected])
                raw,available=features.transitions(batch,bank.directory);assert available.all()
                lengths=torch.full((len(selected),),min(start+1,11),device='cuda',dtype=torch.long)
                p=model.generate(batch['task'],batch['history'],batch['previous_actions'],lengths,start=start,boundary_features=raw)
                observed,memory,previous,codes=model.encode_sequence(p,batch['prefix_codes'])
                before=observed[:,-1].argmax(-1)
                f=model.observation(model.embed(batch['successors'])).flatten(0,1)
                nm,nf,_=model.advance(memory.repeat_interleave(17,0),previous.repeat_interleave(17,0),codes.repeat_interleave(17,0),f,batch['successors'].flatten(0,1))
                nf=F.normalize(nf+model.memory_projection(nm),dim=-1)
                defs=F.normalize(p['definitions'],dim=-1).repeat_interleave(17,0)
                after=torch.einsum('bd,bkd->bk',nf,defs).reshape(len(selected),17,128).argmax(-1)
                probs=p['transitions'][torch.arange(len(selected),device='cuda'),before]
                pred=probs.argmax(-1);valid=pred==after
                counts['states']+=len(selected);counts['branches']+=int(valid.numel());counts['correct']+=int(valid.sum())
                counts['movement_branches']+=int(valid[:,1:].numel());counts['movement_correct']+=int(valid[:,1:].sum())
                counts['STOP_correct']+=int(valid[:,0].sum())
                conf=probs.max(-1).values
                for tag,mask in [('correct',valid),('incorrect',~valid)]:confidence[tag].extend(conf[mask].cpu().tolist())
                ids=before.cpu().tolist();outs=after.cpu().tolist();preds=pred.cpu().tolist()
                for key,s,targets,predictions in zip(keys,ids,outs,preds):
                    programs[key][s]+=1
                    for action,(target,guess) in enumerate(zip(targets,predictions)):
                        rules[(key,s,action)][target]+=1
                for i,(name,t) in enumerate(selected):
                    if start==0 and offset==0:matched_steps.append(dict(bank=bank.directory.name,episode=name,observed_state=ids[i],actual_successors=outs[i],predictions=preds[i]))
        stats[bank.directory.name]=dict(counts,mean_probability_correct=float(np.mean(confidence['correct'])),mean_probability_incorrect=float(np.mean(confidence['incorrect'])))
        print(json.dumps(dict(bank=bank.directory.name,statistics=stats[bank.directory.name],seconds=time.time()-began)),flush=True)
    total=sum(sum(v.values()) for v in rules.values());best=sum(max(v.values()) for v in rules.values())
    conflicting={k:v for k,v in rules.items() if len(v)>1}
    examples=[]
    for (key,s,a),counts in sorted(conflicting.items(),key=lambda x:-sum(x[1].values()))[:30]:
        examples.append(dict(episode=key[0],start=key[1],state=s,action=a,observed_successor_counts=dict(counts)))
    result=dict(training_only=True,no_new_rollouts=True,no_optimizer_step=True,checkpoint=str(path),sha256=sha(path),
        per_bank=stats,programs=len(programs),rule_rows=len(rules),observed_branches=total,
        contradictory_rule_rows=len(conflicting),branches_in_contradictory_rows=sum(sum(v.values()) for v in conflicting.values()),
        best_possible_deterministic_accuracy_on_recorded_states=best/total,
        current_mean_accuracy=sum(v['correct'] for v in stats.values())/total,
        conflicts=examples,initial_observed_branches=matched_steps,seconds=time.time()-began)
    output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['initial_observed_branches','conflicts']}),flush=True)

if __name__=='__main__':main()
