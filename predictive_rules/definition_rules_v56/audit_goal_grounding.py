"""Train-only diagnosis of generated goals against measured successful images.

No state is assigned a human meaning. Comparing a demonstration's first view
to its last is an alias diagnostic, not an assertion that every earlier view
was physically outside the success radius. No validation episodes are used.
"""
from collections import defaultdict,Counter
import json,random,re,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
sys.path.insert(0,str(ROOT/'src_temporal_symbols_v55'))
from model import FactoredWorld,Config,hard
from data import sha

def mean(items):return sum(items)/len(items) if items else None

@torch.inference_mode()
def main():
    torch.set_num_threads(2);began=time.time();rng=random.Random(561018)
    path=ROOT/'diagnostic_temporal_symbols_v55/source.pt'
    commit=json.loads(path.with_suffix('.COMMITTED.json').read_text());assert sha(path)==commit['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).eval();model.load_v54(saved['model']);del saved
    manifest=json.loads((ROOT/'run_action_grounded_v33/manifest.json').read_text())
    source=Path(manifest['data']['source']);rows_file=source/'training_rows.json'
    assert sha(rows_file)==manifest['data']['rows_sha256']
    rows=json.loads(rows_file.read_text());raw=np.load(ROOT/'prequant_visual_features_v18/features.npy',mmap_mode='r')
    episodes=defaultdict(list)
    for i,row in enumerate(rows):
        match=re.search(r'P25_train_(\d+)_',row.get('source_episode_id',row['episode_id']))
        assert match and 'layout_'+match.group(1) in manifest['data']['train_layouts']
        episodes[row['episode_id']].append(i)
    names=sorted(episodes);selected=rng.sample(names,min(256,len(names)))
    paired={}
    for task in ('chair','table'):
        pool=[name for name in names if rows[episodes[name][0]]['category']==task]
        paired[task]=pool
    results=[];context_signatures=[]
    for begin in range(0,len(selected),32):
        batch=selected[begin:begin+32];n=len(batch);length=34
        x=np.zeros((n,length,49,128),np.float16);previous=np.full((n,length),17,np.int64)
        end=[];tasks=[];other_success=[];other_ids=[]
        for b,name in enumerate(batch):
            idx=episodes[name];count=len(idx);assert count<=33 and rows[idx[-1]]['label']==0
            x[b,:count]=raw[idx];x[b,count]=x[b,count-1]
            previous[b,1:count+1]=[int(rows[i]['label']) for i in idx]
            task=rows[idx[0]]['category'];tasks.append(['chair','table'].index(task));end.append(count)
            candidate=rng.choice(paired[task]);
            while candidate==name:candidate=rng.choice(paired[task])
            other_success.append(raw[episodes[candidate][-1]].copy());other_ids.append(candidate)
        x=torch.from_numpy(x);previous=torch.from_numpy(previous);task=torch.tensor(tasks);end=torch.tensor(end);ix=torch.arange(n)
        global_observations=hard(model.observe(x))
        contexts=model.context_sequence(x,global_observations,previous,task)
        for boundary_label,boundary in [('initial',torch.zeros_like(end)),('last_before_stop',(end-1).clamp_min(0))]:
            context=contexts[ix,boundary];logits,z,_=model.observation_encoding(x,context)
            obs=hard(logits);memlogits,_,_=model.memory_sequence(obs,previous,z)
            gm,go=model.goal_logits(context,task);mi=gm.argmax(-1);oi=go.argmax(-1)
            true_m=memlogits[ix,end].argmax(-1);true_o=logits[ix,end].argmax(-1)
            start_o=logits[ix,0].argmax(-1)
            alternative=model.observe(torch.tensor(np.stack(other_success))[:,None],context)[:,0].argmax(-1)
            sm=hard(memlogits[ix,(end-1).clamp_min(0)])[:,None]
            so=obs[ix,(end-1).clamp_min(0)][:,None]
            pm,po=model.transition(context,sm,so,torch.zeros(n,1,dtype=torch.long))
            for b,name in enumerate(batch):
                category=rows[episodes[name][0]]['category']
                match=re.search(r'_(near|medium|far)_',name)
                results.append(dict(episode=name,boundary=boundary_label,category=category,
                    distance_group=match.group(1) if match else 'unknown',demonstrated_steps=int(end[b]),
                    predicted_goal_memory=int(mi[b]),predicted_goal_observation=int(oi[b]),
                    actual_terminal_memory=int(true_m[b]),actual_terminal_observation=int(true_o[b]),
                    goal_memory_correct=bool(mi[b]==true_m[b]),goal_observation_correct=bool(oi[b]==true_o[b]),
                    goal_joint_correct=bool((mi[b]==true_m[b])&(oi[b]==true_o[b])),
                    first_last_observation_alias=bool(start_o[b]==true_o[b]),
                    predicted_goal_matches_initial_observation=bool(start_o[b]==oi[b]),
                    another_successful_same_task_alias=bool(alternative[b]==true_o[b]),
                    stop_memory_correct=bool(pm[b,0].argmax()==true_m[b]),
                    stop_observation_correct=bool(po[b,0].argmax()==true_o[b]),
                    actual_terminal_reaches_generated_goal_in_one_stop=bool((pm[b,0].argmax()==mi[b])&(true_o[b]==oi[b])),
                    other_success_episode=other_ids[b]))
    fields=['goal_memory_correct','goal_observation_correct','goal_joint_correct','first_last_observation_alias',
        'predicted_goal_matches_initial_observation','another_successful_same_task_alias','stop_memory_correct',
        'stop_observation_correct','actual_terminal_reaches_generated_goal_in_one_stop']
    def stats(values):return dict(episodes=len(values),**{k:mean([r[k] for r in values]) for k in fields})
    summary={b:stats([r for r in results if r['boundary']==b]) for b in ('initial','last_before_stop')}
    groups={}
    for b in summary:
        for field in ('category','distance_group'):
            for value in sorted({r[field] for r in results}):
                groups[b+'/'+field+':'+value]=stats([r for r in results if r['boundary']==b and r[field]==value])
    result=dict(time=time.time(),seconds=time.time()-began,source=commit,training_only=True,new_simulator_rollouts=0,
        original_dev80_used=False,finalfixed80_used=False,gradients=0,summary=summary,by_group=groups,rows=results,
        limitation='These successful demonstrations are training data. Initial/terminal alias is descriptive, not a physically false-stop label. Cross-episode terminal-code equality is not required by the task; low agreement diagnoses whether goal images are represented coherently, not a correctness violation.')
    out=ROOT/'diagnostic_definition_rules_v56/goal_grounding.json';out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','by_group')}),flush=True)

if __name__=='__main__':main()
