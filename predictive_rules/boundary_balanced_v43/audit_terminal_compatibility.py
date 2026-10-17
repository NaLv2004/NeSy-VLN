"""Score successful training endpoints without changing goals or executing actions."""
from collections import defaultdict
import json,os,random,time
from pathlib import Path
import numpy as np
import torch
from model import FactoredWorld,Config,hard
from data import ROOT,sha

@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.15)
    torch.backends.cuda.matmul.allow_tf32=True;began=time.time()
    out=ROOT/'diagnostic_terminal_compatibility_v43';out.mkdir(exist_ok=True)
    assert not (out/'report.json').exists(),'Immutable audit already complete'
    source=ROOT/'diagnostic_sparse_support_v44';commit=json.loads((source/'source.COMMITTED.json').read_text())
    assert sha(source/'source.pt')==commit['sha256']
    saved=torch.load(source/'source.pt',map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).cuda().eval();model.load_state_dict(saved['model']);c=model.cfg
    rows=json.loads((Path(saved['manifest']['data']['source'])/'training_rows.json').read_text())
    raw=np.load(ROOT/'prequant_visual_features_v18/features.npy',mmap_mode='r')
    episodes=defaultdict(list);canonical={}
    for i,row in enumerate(rows):
        episodes[row['episode_id']].append(i)
        canonical[row['episode_id']]=row.get('source_episode_id',row['episode_id'])
    unique={}
    for name in sorted(episodes):unique.setdefault(canonical[name],name)
    groups=defaultdict(list)
    for origin,name in unique.items():
        row=rows[episodes[name][0]]
        category=row['category'];distance=next((s for s in ('near','medium','far') if '_'+s+'_' in origin),'unknown')
        groups[(category,distance)].append(name)
    rng=random.Random(461017);chosen=[]
    for key,names in sorted(groups.items()):chosen.extend(rng.sample(names,min(40,len(names))))
    records=[]
    for name in chosen:
        indices=episodes[name];n=len(indices);labels=[int(rows[i]['label']) for i in indices];assert labels[-1]==0
        x=torch.tensor(np.concatenate([raw[indices],raw[indices[-1:]]],0),device='cuda')[None]
        previous=torch.tensor([[17]+labels],device='cuda');task=torch.tensor([['chair','table'].index(rows[indices[0]]['category'])],device='cuda')
        logits,z,_=model.observation_encoding(x);o=hard(logits);lm,_,_=model.memory_sequence(o,previous,z);m=hard(lm)
        contexts=model.context_sequence(x,o,previous,task)
        memory_codes=lm.argmax(-1)[0];observation_codes=logits.argmax(-1)[0]
        record=dict(episode=name,canonical=canonical[name],actions=n,
            unchanged_image_on_stop=bool(np.array_equal(raw[indices[-1]],x[0,-1].cpu().numpy())),
            actual_memory_changes_on_stop=bool(memory_codes[-1]!=memory_codes[-2]),
            actual_observation_changes_on_stop=bool(observation_codes[-1]!=observation_codes[-2]))
        for label,boundary in [('initial',0),('last_five',max(0,n-5)),('fresh_terminal',n-1)]:
            context=contexts[:,boundary];gm,go=model.goal_logits(context,task)
            pm,po=model.transition(context,m[:,n-1],o[:,n-1],torch.tensor([0],device='cuda'))
            goal_m=int(gm.argmax(-1));goal_o=int(go.argmax(-1));next_m=int(pm.argmax(-1))
            record[label]=dict(goal_memory_matches_endpoint=goal_m==int(memory_codes[-1]),
                goal_observation_matches_endpoint=goal_o==int(observation_codes[-1]),
                predicted_stop_memory_matches_endpoint=next_m==int(memory_codes[-1]),
                stop_can_reach_generated_goal=(next_m==goal_m and goal_o==int(observation_codes[-2])),
                goal_m=goal_m,goal_o=goal_o,predicted_stop_m=next_m)
        # Replay the RECORDED successful actions with independent predicted M,
        # current O feedback, and precisely the normal learned renewal rule.
        memory=int(memory_codes[0]);context=contexts[:,0];age=0;predicted_observation=None;renewals=1
        for t,action in enumerate(labels):
            observed=int(observation_codes[t])
            om=torch.nn.functional.one_hot(torch.tensor([observed],device='cuda'),64).float()
            mm=torch.nn.functional.one_hot(torch.tensor([memory],device='cuda'),32).float()
            if t:
                pred=torch.nn.functional.one_hot(torch.tensor([predicted_observation],device='cuda'),64).float()
                renew=age>=5 and float(model.renewal_logits(mm,om,pred,torch.tensor([age],device='cuda')).sigmoid())>=.5
                if renew:
                    memory=int(memory_codes[t]);mm=m[:,t];context=contexts[:,t];age=0;renewals+=1
            gm,go=model.goal_logits(context,task)
            pm,po=model.transition(context,mm,om,torch.tensor([action],device='cuda'))
            memory=int(pm.argmax(-1));predicted_observation=int(po.argmax(-1));age+=1
        record['teacher_replay']=dict(renewals=renewals,predicted_memory_correct=memory==int(memory_codes[-1]),
            stop_reaches_generated_goal=memory==int(gm.argmax(-1)) and int(observation_codes[-1])==int(go.argmax(-1)))
        records.append(record)
    metrics={key:float(np.mean([r[key] for r in records])) for key in ('actual_memory_changes_on_stop','actual_observation_changes_on_stop')}
    for condition in ('initial','last_five','fresh_terminal','teacher_replay'):
        metrics[condition]={key:float(np.mean([r[condition][key] for r in records])) for key,value in records[0][condition].items() if isinstance(value,bool)}
    report=dict(time=time.time(),source_commit=commit,training_only=True,episodes=len(records),metrics=metrics,records=records,
        seconds=time.time()-began,policy_modified=False,new_environment_rollouts=0,
        limitation='Successful teacher paths only. Selected-row network calls here score the same exported function; no actions are selected by this diagnostic.')
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'}))
if __name__=='__main__':main()
