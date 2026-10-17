"""Train-only paired comparison: O-terminal agreement vs joint M/O agreement.

The neural modules still only ground/generate an anonymous symbolic program.
All future terminal histories below are supervision, never runtime inputs.
"""
import copy
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
sys.path.insert(0,str(ROOT/'src_goal_equivalence_v57'))
from model import FactoredWorld,Config,hard
from objectives import objective
from data import Corpus,sha


def endpoints(model,data):
    x=data['features'];ix=torch.arange(len(x));end=data['end'];task=data['task']
    context=model.context_sequence(x,hard(model.observe(x)),data['previous'],task)[ix,data['start']]
    own,z,_=model.observation_encoding(x,context)
    own_m,_,_=model.memory_sequence(hard(own),data['previous'],z)
    other,az,_=model.observation_encoding(data['alternate_features'],context)
    other_m,_,_=model.memory_sequence(hard(other),data['alternate_previous'],az)
    alt_end=data['alternate_end']
    lm=own_m[ix,end].log_softmax(-1);am=other_m[ix,alt_end].log_softmax(-1)
    pair=-torch.logsumexp(lm+am,-1).mean()
    gm,go=model.goal_logits(context,task);gmi=gm.argmax(-1);goi=go.argmax(-1)
    oi=own[ix,end].argmax(-1);mi=own_m[ix,end].argmax(-1)
    aoi=other[ix,alt_end].argmax(-1);ami=other_m[ix,alt_end].argmax(-1)
    # Independent complete STOP-preimage query over every integer memory ID.
    c=model.cfg
    candidates=F.one_hot(torch.arange(c.memories)[None].expand(len(x),-1),c.memories).float()
    goals=F.one_hot(goi,c.observations).float()[:,None].expand(-1,c.memories,-1)
    nm,_=model.transition(context,candidates,goals,torch.zeros(len(x),c.memories,dtype=torch.long))
    am_before=hard(other_m[ix,alt_end-1]);ao_before=hard(other[ix,alt_end-1])
    sm,so=model.transition(context,am_before,ao_before,torch.zeros(len(x),dtype=torch.long))
    terminal_mask=torch.arange(x.shape[1])[None]<=end[:,None]
    memory_ids=own_m.argmax(-1)
    count=F.one_hot(memory_ids[terminal_mask],c.memories).float().mean(0)
    metrics=dict(memory_pair_loss=pair.detach(),
        memory_pair_agreement=(mi==ami).float().mean(),
        observation_pair_agreement=(oi==aoi).float().mean(),
        own_joint_goal_agreement=((mi==gmi)&(oi==goi)).float().mean(),
        alternate_joint_goal_agreement=((ami==gmi)&(aoi==goi)).float().mean(),
        alternate_stop_reaches_goal=((sm.argmax(-1)==gmi)&(so.argmax(-1)==goi)).float().mean(),
        goal_has_stop_preimage=(nm.argmax(-1)==gmi[:,None]).any(-1).float().mean(),
        first_joint_goal_alias=((own[:,0].argmax(-1)==goi)&(own_m[:,0].argmax(-1)==gmi)).float().mean(),
        first_terminal_memory_alias=(own_m[:,0].argmax(-1)==mi).float().mean(),
        hard_memory_perplexity=(-(count*count.clamp_min(1e-9).log()).sum()).exp(),
        mean_memory_ids_per_program=torch.tensor([len(memory_ids[b,:int(end[b])+1].unique()) for b in range(len(x))],dtype=torch.float32).mean())
    return pair,metrics


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2);torch.manual_seed(591017);os.nice(10);began=time.time()
    out=ROOT/'diagnostic_joint_endpoint_v59';out.mkdir(exist_ok=True)
    assert not (out/'result.json').exists(),'Do not repeat completed pilot'
    source=ROOT/'run_goal_equivalence_v57/checkpoint_001000.pt'
    commit=json.loads(source.with_suffix('.COMMITTED.json').read_text())
    assert commit['sha256']=='58abc7f6a46587f2312412b1848c1e84dc64a374ba725b81f51dd8c0bca880bf'
    pinned=out/'source.pt'
    if not pinned.exists():os.link(source,pinned)
    assert sha(pinned)==commit['sha256']
    (out/'source.COMMITTED.json').write_text(json.dumps(dict(commit,source_path=str(source)),indent=2))
    saved=torch.load(pinned,map_location='cpu',weights_only=False)
    initial=FactoredWorld(Config(**saved['model_config']));initial.load_state_dict(saved['model'],strict=True)
    del saved
    models={name:copy.deepcopy(initial) for name in ('observation_only_control','joint_endpoint')}
    del initial
    corpus=Corpus(branch_root=str(ROOT/'onpolicy_successors_v54'))
    rng=random.Random(591017);names=list(corpus.names);rng.shuffle(names)
    parts={'holdout':names[:128],'fit':names[128:]}
    assert not set(parts['fit'])&set(parts['holdout'])
    pools={part:{task:[n for n in group if corpus.rows[corpus.episodes[n][0]]['category']==task]
        for task in ('chair','table')} for part,group in parts.items()}
    def sample(part,rng):
        corpus.names=parts[part]
        data=corpus.sample(rng,8,4,'cpu');size=len(data['features'])
        ax=np.zeros((size,34,49,128),np.float16);prev=np.full((size,34),17,np.int64);ends=[]
        for b,task in enumerate(data['task']):
            for attempt in range(100):
                name=rng.choice(pools[part][('chair','table')[int(task)]])
                ids=corpus.episodes[name];n=len(ids)
                if not np.array_equal(corpus.raw[ids[-1]],data['features'][b,data['end'][b]].numpy()):break
            else:raise RuntimeError('Distinct terminal not found')
            ax[b,:n]=corpus.raw[ids];ax[b,n]=ax[b,n-1]
            prev[b,1:n+1]=[corpus.rows[i]['label'] for i in ids];ends.append(n)
        data.update(alternate_features=torch.tensor(ax),alternate_previous=torch.tensor(prev),alternate_end=torch.tensor(ends))
        data['other_success']=data['alternate_features'][torch.arange(size),data['alternate_end']]
        return data
    probes={part:[sample(part,random.Random(591100+i+100*(part=='holdout'))) for i in range(8)] for part in parts}
    optimizers={}
    for name,model in models.items():
        groups=[[],[]]
        for key,p in model.named_parameters():
            if p.requires_grad:groups[0 if key.startswith(('visual.','observation_definitions','memory.','memory_definitions')) else 1].append(p)
        optimizers[name]=torch.optim.AdamW([dict(params=groups[0],lr=2e-5),dict(params=groups[1],lr=1e-4)],weight_decay=1e-4)
    history=[]
    fields=('loss','single_step_observation_accuracy','single_step_memory_accuracy','rollout_observation_accuracy',
        'rollout_memory_accuracy','branch_observation_accuracy','branch_changed_observation_accuracy','inverse_accuracy',
        'goal_memory_accuracy','goal_observation_accuracy','hard_observation_perplexity')
    @torch.no_grad()
    def evaluate(update):
        results={}
        for name,model in models.items():
            model.eval();results[name]={}
            for part,batches in probes.items():
                rows=[]
                for i,data in enumerate(batches):
                    torch.manual_seed(591200+i);data['world_updates_encoder']=False
                    _,base=objective(model,data);_,extra=endpoints(model,data)
                    rows.append({**{key:float(base[key]) for key in fields},**{key:float(value) for key,value in extra.items()}})
                results[name][part]={key:sum(r[key] for r in rows)/len(rows) for key in rows[0]}
            model.train()
        row=dict(update=update,seconds=time.time()-began,metrics=results);history.append(row)
        (out/'progress.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
    evaluate(0)
    sampling=random.Random(591019)
    for update in range(1,97):
        data=sample('fit',sampling);data['world_updates_encoder']=update%5==0
        for name,model in models.items():
            torch.manual_seed(591500+update)
            optimizers[name].zero_grad(set_to_none=True)
            loss,_=objective(model,data)
            if name=='joint_endpoint':
                memory_pair,_=endpoints(model,data);loss=loss+memory_pair
            assert torch.isfinite(loss);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm)
            optimizers[name].step()
        if update%32==0:evaluate(update)
    result=dict(time=time.time(),seconds=time.time()-began,pid=os.getpid(),source=commit,
        source_path=str(pinned),updates=96,history=history,fit_episodes=parts['fit'],heldout_training_episodes=parts['holdout'],
        device='CPU',threads=2,live_training_changed=False,official_dev80_used=False,finalfixed80_used=False,new_rollouts=0,
        adaptation='Add successful terminal-memory agreement under the same fixed boundary definitions. Match the existing O-terminal control in data, seed, optimizer and initialization.',
        runtime_changes=None,no_hand_assigned_symbol_ids=True,no_neural_action_policy=True,
        caveat='Source saw this broader training corpus. The holdout is only disjoint for additional fitting. Alias metrics are not physical false-stop counts; this is not navigation validation.')
    torch.save(dict(model=models['joint_endpoint'].state_dict(),config=models['joint_endpoint'].export_config(),source=commit),out/'probe.pt')
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('history','fit_episodes','heldout_training_episodes')}),flush=True)

if __name__=='__main__':main()
