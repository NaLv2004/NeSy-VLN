"""Bounded training-only pilot. No navigation validation, no live-run mutation."""
from collections import Counter,defaultdict
import json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import FactoredWorld,Config,hard
from data import Corpus,ROOT,sha
from planner import Program

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.20)
    torch.backends.cuda.matmul.allow_tf32=True;torch.manual_seed(451017);rng=random.Random(451017)
    out=ROOT/'diagnostic_state_sufficiency_v45';out.mkdir(exist_ok=True)
    assert not (out/'report.json').exists(),'Immutable pilot already finished'
    source=ROOT/'diagnostic_sparse_support_v44';descriptor=json.loads((source/'dataset.json').read_text())
    commit=json.loads((source/'source.COMMITTED.json').read_text());assert sha(source/'source.pt')==commit['sha256']
    saved=torch.load(source/'source.pt',map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).cuda();missing,unexpected=model.load_state_dict(saved['model'],strict=False)
    assert not unexpected and all(k.startswith('effect_aux.') for k in missing)
    c=model.cfg;began=time.time();excluded=set(descriptor['heldout_training_episodes'])
    corpus=Corpus(branch_root=ROOT/'onpolicy_successors_v40')
    corpus.names=[name for name in corpus.names if corpus.rows[corpus.episodes[name][0]].get('source_episode_id',name) not in excluded]
    for key in ('old_branch_positions','onpolicy_positions','branch_positions'):
        setattr(corpus,key,[p for p in getattr(corpus,key) if corpus.branches[p[0]][0]['episode'] not in excluded])
    for label,positions in [('original',corpus.old_branch_positions),('onpolicy',corpus.onpolicy_positions)]:
        corpus.phase_pools[label]=([p for p in positions if p[1]==0],[p for p in positions if 1<=p[1]<5],[p for p in positions if p[1]>=5])
    assert all(corpus.branches[i][0]['episode'] not in excluded for i,t in corpus.branch_positions)
    cases=[]
    for file in sorted((ROOT/'onpolicy_successors_v40/counterfactual_onpolicy').glob('*.json')):
        record=json.loads(file.read_text())
        if record['episode'] not in excluded:continue
        with np.load(file.with_suffix('.npz')) as data:
            n=len(data['features']);pos=sorted(set([0,min(1,n-1),min(5,n-1),n-1]))
            cases.append(dict(features=data['features'].copy(),successors=data['successor_features'][pos].copy(),
                previous=np.asarray([17]+data['recorded_actions'][:n-1].tolist()),positions=pos))
    assert len(cases)==24
    @torch.no_grad()
    def measure():
        model.eval();counts=Counter();obs_labels=[];joint_rows=[]
        for item in cases:
            x=torch.tensor(item['features'],device='cuda')[None];previous=torch.tensor(item['previous'],device='cuda')[None]
            ol,z,_=model.observation_encoding(x);o=hard(ol);ml,_,_=model.memory_sequence(o,previous,z);m=hard(ml)
            contexts=model.context_sequence(x,o,previous,torch.tensor([0],device='cuda'))
            pos=item['positions'];future=torch.tensor(item['successors'],device='cuda')
            next_o,znext,_=model.observation_encoding(future);target_o=next_o.argmax(-1)
            m0=m[0,pos,None].expand(-1,17,-1);o0=o[0,pos,None].expand(-1,17,-1)
            actions=torch.arange(17,device='cuda')[None].expand(len(pos),-1)
            next_m,_=model.memory_step(m0,hard(next_o),actions,znext);target_m=next_m.argmax(-1)
            context=contexts[0,[max(0,t-5) for t in pos]]
            pm,po=model.transition(context,m0,o0,actions)
            am,ao=model.effect_aux(torch.cat([m0,o0,F.one_hot(actions,17).float()],-1)).split([32,64],-1)
            inverse=model.inverse(torch.cat([o0,hard(next_o),F.one_hot(torch.zeros_like(actions),10).float()],-1)).argmax(-1)
            eq=(future[:,:,None].float()-future[:,None,:].float()).square().mean((-1,-2))<1e-10
            inverse_valid=eq.gather(-1,inverse[...,None])[...,0]
            for key,val in [('world_observation',po.argmax(-1)==target_o),('world_memory',pm.argmax(-1)==target_m),
                ('context_free_observation',ao.argmax(-1)==target_o),('context_free_memory',am.argmax(-1)==target_m),('inverse_equivalent_action',inverse_valid)]:
                counts[key]+=int(val.sum())
            counts['outcomes']+=target_o.numel();obs_labels.extend(o[0,pos].argmax(-1).tolist())
            joint_rows.extend(zip(m0.argmax(-1).flatten().tolist(),o0.argmax(-1).flatten().tolist(),actions.flatten().tolist(),target_o.flatten().tolist()))
        histogram=np.bincount(obs_labels,minlength=64);p=histogram/histogram.sum();p=p[p>0]
        table=defaultdict(Counter)
        for m,o,a,n in joint_rows:table[(m,o,a)][n]+=1
        return dict(outcomes=counts['outcomes'],**{k+'_accuracy':counts[k]/counts['outcomes'] for k in counts if k!='outcomes'},
            observation_codes_used=int((histogram>0).sum()),observation_perplexity=float(np.exp(-(p*np.log(p)).sum())),
            empirical_context_free_accuracy_ceiling=sum(max(v.values()) for v in table.values())/len(joint_rows))
    before=measure();print(json.dumps(dict(phase='before',metrics=before)),flush=True)
    # Warm up only the new classifier on fixed TRAINING codes so random head
    # gradients cannot initially scramble an already learned representation.
    with np.load(source/'fixed_training_cases.npz') as fixed:
        fit=~fixed['heldout'];m=torch.tensor(fixed['memory'][fit],device='cuda');o=torch.tensor(fixed['observation'][fit],device='cuda')
        labels_m=torch.tensor(fixed['target_m'][fit],device='cuda');labels_o=torch.tensor(fixed['target_o'][fit],device='cuda')
    a=F.one_hot(torch.arange(17,device='cuda'),17).float()[None].expand(len(m),-1,-1);inputs=torch.cat([m,o,a],-1)
    warm=torch.optim.AdamW(model.effect_aux.parameters(),lr=5e-4)
    for step in range(200):
        indices=torch.randint(len(inputs),(64,),device='cuda');lm,lo=model.effect_aux(inputs[indices]).split([32,64],-1)
        loss=F.cross_entropy(lm.flatten(0,1),labels_m[indices].flatten())+F.cross_entropy(lo.flatten(0,1),labels_o[indices].flatten())
        warm.zero_grad(set_to_none=True);loss.backward();warm.step()
    representation=('visual.','observation_definitions','memory.','memory_definitions')
    optimizer=torch.optim.AdamW([
        dict(params=[p for n,p in model.named_parameters() if n.startswith(representation)],lr=2e-5),
        dict(params=[p for n,p in model.named_parameters() if not n.startswith(representation)],lr=1e-4)],weight_decay=1e-4)
    history=[dict(update=0,metrics=measure())]
    for update in range(1,301):
        model.train();data=corpus.sample(rng,32,32,'cuda');data['world_updates_encoder']=update%5==0
        loss,metrics=model(data);assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        assert torch.isfinite(norm);optimizer.step()
        if update%100==0:
            row=dict(update=update,metrics=measure(),seconds=time.time()-began)
            history.append(row);print(json.dumps(row),flush=True)
    # During rule compilation the new classifier must remain unused.
    def forbidden(*_):raise RuntimeError('Training-only effect classifier entered runtime')
    handle=model.effect_aux.register_forward_pre_hook(forbidden)
    with torch.no_grad():Program(**model.compile(torch.randn(1,128,device='cuda'),torch.tensor([0],device='cuda'))).validate()
    handle.remove()
    torch.save(dict(model=model.state_dict(),model_config=model.export_config(),source_commit=commit,diagnostic_only=True),out/'candidate_000300.pt')
    report=dict(time=time.time(),source_commit=commit,before=before,history=history,seconds=time.time()-began,
        excluded_pilot_episodes=sorted(excluded),fit_demonstrations=len(corpus.names),fit_branch_positions=len(corpus.branch_positions),
        gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,live_training_changed=False,
        runtime_effect_head_calls=0,validation_used=False,
        limitation='Held-out only for this pilot; source checkpoint saw the broader training corpus. Codes are learned jointly, so prediction accuracy alone cannot establish physical usefulness.')
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(complete=True,seconds=report['seconds'])),flush=True)
if __name__=='__main__':main()
