"""Check endpoint alignment did not simply label every current view a goal."""
import json, random, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
sys.path.insert(0,str(ROOT/'src_definition_rules_v56'))
from model import FactoredWorld,Config,hard
from data import sha

@torch.inference_mode()
def main():
    torch.set_num_threads(2);began=time.time();rng=random.Random(571018)
    out=ROOT/'diagnostic_goal_equivalence_v57';pilot=json.loads((out/'result.json').read_text())
    manifest=json.loads((ROOT/'run_action_grounded_v33/manifest.json').read_text())
    path=Path(manifest['data']['source'])/'training_rows.json';assert sha(path)==manifest['data']['rows_sha256']
    rows=json.loads(path.read_text());raw=np.load(ROOT/'prequant_visual_features_v18/features.npy',mmap_mode='r')
    episodes=defaultdict(list)
    for i,r in enumerate(rows):episodes[r['episode_id']].append(i)
    names=pilot['heldout_training_episodes'];pairs=[]
    for name in names:
        category=rows[episodes[name][0]]['category']
        pairs.append(rng.choice([n for n in names if n!=name and rows[episodes[n][0]]['category']==category]))
    def load_batch(group):
        x=np.zeros((len(group),34,49,128),np.float16);prev=np.full((len(group),34),17,np.int64)
        ends=[];tasks=[]
        for b,name in enumerate(group):
            ids=episodes[name];n=len(ids);assert n<=33 and rows[ids[-1]]['label']==0
            x[b,:n]=raw[ids];x[b,n]=x[b,n-1];prev[b,1:n+1]=[rows[i]['label'] for i in ids]
            ends.append(n);tasks.append(['chair','table'].index(rows[ids[0]]['category']))
        return torch.tensor(x),torch.tensor(prev),torch.tensor(ends),torch.tensor(tasks)
    all_results={}
    for variant in ('source','aligned'):
        if variant=='source':
            p=ROOT/'diagnostic_temporal_symbols_v55/source.pt';assert sha(p)==pilot['source']['sha256']
            saved=torch.load(p,map_location='cpu',weights_only=False)
            model=FactoredWorld(Config(**saved['model_config']))
            head=torch.load(ROOT/'diagnostic_definition_rules_v56/head.pt',map_location='cpu',weights_only=False)
            model.load_initial(saved['model'],head['head']);del head
        else:
            saved=torch.load(out/'probe.pt',map_location='cpu',weights_only=False)
            model=FactoredWorld(Config(**saved['config']));model.load_state_dict(saved['model'])
        del saved;model.eval();records=[]
        for b in range(0,len(names),16):
            group=names[b:b+16];other=pairs[b:b+16]
            x,prev,end,task=load_batch(group);ax,aprev,aend,atask=load_batch(other);assert torch.equal(task,atask)
            ix=torch.arange(len(x));context=model.context_sequence(x,hard(model.observe(x)),prev,task)[:,0]
            lo,z,_=model.observation_encoding(x,context);o=hard(lo);lm,_,_=model.memory_sequence(o,prev,z)
            alo,az,_=model.observation_encoding(ax,context);ao=hard(alo);alm,_,_=model.memory_sequence(ao,aprev,az)
            gm,go=model.goal_logits(context,task);gmi=gm.argmax(-1);goi=go.argmax(-1)
            mi=lm[ix,end].argmax(-1);oi=lo[ix,end].argmax(-1)
            ami=alm[ix,aend].argmax(-1);aoi=alo[ix,aend].argmax(-1)
            am_before=hard(alm[ix,aend-1]);ao_before=ao[ix,aend-1]
            nextm,nexto=model.transition(context,am_before,ao_before,torch.zeros(len(x),dtype=torch.long))
            for i,name in enumerate(group):
                records.append(dict(episode=name,paired_episode=other[i],
                    own_goal_joint=bool((mi[i]==gmi[i])&(oi[i]==goi[i])),
                    alternate_goal_observation=bool(aoi[i]==goi[i]),
                    alternate_goal_memory=bool(ami[i]==gmi[i]),
                    alternate_goal_joint=bool((ami[i]==gmi[i])&(aoi[i]==goi[i])),
                    alternate_stop_reaches_goal=bool((nextm[i].argmax()==gmi[i])&(nexto[i].argmax()==goi[i])),
                    first_final_observation_alias=bool(lo[i,0].argmax()==oi[i]),
                    first_observation_is_goal=bool(lo[i,0].argmax()==goi[i]),
                    first_joint_is_goal=bool((lo[i,0].argmax()==goi[i])&(lm[i,0].argmax()==gmi[i]))))
        metrics={k:sum(r[k] for r in records)/len(records) for k in records[0] if k not in ('episode','paired_episode')}
        all_results[variant]=dict(episodes=len(records),metrics=metrics,rows=records)
    result=dict(time=time.time(),seconds=time.time()-began,results=all_results,training_only=True,
        original_dev80_used=False,finalfixed80_used=False,live_training_modified=False,
        caveat='Start/goal alias is descriptive, not a physical false-stop label. Alternate histories are used ONLY for this offline grounding audit, not runtime correction.')
    (out/'grounding_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{'episodes':v['episodes'],'metrics':v['metrics']} for k,v in all_results.items()}),flush=True)

if __name__=='__main__':main()
