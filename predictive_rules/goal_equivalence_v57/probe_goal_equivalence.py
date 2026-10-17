"""Train-only positive endpoint agreement, with the full v56 world objective.

This is an isolated CPU experiment. Terminal pairs are measured successful
demonstrations for the same requested category, not manually assigned symbol
IDs. Nonterminal frames are NOT labelled unsuccessful. No policy/planner is
added or changed and no official validation observations are read.
"""
import json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
sys.path.insert(0,str(ROOT/'src_definition_rules_v56'))
from model import FactoredWorld, Config, hard
from data import Corpus, sha
from objectives import objective


def endpoint_agreement(model,data):
    x=data['features'];idx=torch.arange(len(x));end=data['end']
    context=model.context_sequence(x,hard(model.observe(x)),data['previous'],data['task'])[idx,data['start']]
    # All images are interpreted with exactly the same fixed boundary basis.
    own=x[idx,end][:,None]
    pair=torch.cat([own,data['other_success'][:,None]],dim=1)
    logits=model.observe(pair,context)
    logp=logits.log_softmax(-1)
    loss=-torch.logsumexp(logp[:,0]+logp[:,1],dim=-1).mean()
    _,goal=model.goal_logits(context,data['task'])
    ids=logits.argmax(-1);g=goal.argmax(-1)
    return loss,dict(endpoint_pair_agreement=(ids[:,0]==ids[:,1]).float().mean(),
        own_terminal_goal_agreement=(ids[:,0]==g).float().mean(),
        alternate_terminal_goal_agreement=(ids[:,1]==g).float().mean(),
        endpoint_pair_loss=loss.detach())


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2);torch.manual_seed(571017);began=time.time()
    out=ROOT/'diagnostic_goal_equivalence_v57';out.mkdir(exist_ok=True)
    assert not (out/'result.json').exists(),'Do not repeat a completed experiment'
    source=ROOT/'diagnostic_temporal_symbols_v55/source.pt'
    commit=json.loads(source.with_suffix('.COMMITTED.json').read_text());assert sha(source)==commit['sha256']
    saved=torch.load(source,map_location='cpu',weights_only=False)
    head=torch.load(ROOT/'diagnostic_definition_rules_v56/head.pt',map_location='cpu',weights_only=False)
    assert head['source']['sha256']==commit['sha256']
    model=FactoredWorld(Config(**saved['model_config']));model.load_initial(saved['model'],head['head'])
    del saved,head
    corpus=Corpus(branch_root=str(ROOT/'onpolicy_successors_v54'))
    rng=random.Random(571017);names=list(corpus.names);rng.shuffle(names)
    heldout=names[:128];fit=names[128:];assert not set(heldout)&set(fit)
    pools={part:{task:[n for n in group if corpus.rows[corpus.episodes[n][0]]['category']==task]
                 for task in ('chair','table')} for part,group in [('fit',fit),('holdout',heldout)]}
    def sample(part,randomizer):
        corpus.names=fit if part=='fit' else heldout
        data=corpus.sample(randomizer,8,4,'cpu');others=[]
        for b,task in enumerate(data['task']):
            candidates=pools[part][('chair','table')[int(task)]]
            own=data['features'][b,data['end'][b]].numpy()
            for attempt in range(100):
                name=randomizer.choice(candidates);frame=corpus.raw[corpus.episodes[name][-1]]
                if not np.array_equal(own,frame):break
            else:raise RuntimeError('No distinct positive endpoint')
            others.append(frame.copy())
        data['other_success']=torch.tensor(np.stack(others));return data
    # Same batches and same random inverse offsets at each readout.
    probes={part:[sample(part,random.Random(571100+i+(part=='holdout')*100)) for i in range(8)]
            for part in ('fit','holdout')}
    representation={'observation_definitions','memory_definitions'}
    repr_params=[];other_params=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        (repr_params if name in representation or name.startswith(('visual.','memory.')) else other_params).append(p)
    optimizer=torch.optim.AdamW([{'params':repr_params,'lr':2e-5},{'params':other_params,'lr':1e-4}])
    history=[]
    fields=('loss','single_step_observation_accuracy','rollout_observation_accuracy','branch_observation_accuracy',
        'branch_changed_observation_accuracy','inverse_accuracy','hard_observation_perplexity',
        'goal_memory_accuracy','goal_observation_accuracy','mean_codes_per_training_program')
    @torch.no_grad()
    def evaluate(update):
        model.eval();values={}
        for part,batches in probes.items():
            rows=[]
            for i,data in enumerate(batches):
                torch.manual_seed(571200+i);data['world_updates_encoder']=False
                _,metrics=objective(model,data);_,paired=endpoint_agreement(model,data)
                rows.append({**{k:float(metrics[k]) for k in fields},**{k:float(v) for k,v in paired.items()}})
            values[part]={k:sum(r[k] for r in rows)/len(rows) for k in rows[0]}
        record=dict(update=update,seconds=time.time()-began,metrics=values)
        history.append(record);(out/'progress.json').write_text(json.dumps(record,indent=2))
        print(json.dumps(record),flush=True);model.train()
    evaluate(0)
    for update in range(1,129):
        data=sample('fit',rng);data['world_updates_encoder']=update%5==0
        optimizer.zero_grad(set_to_none=True)
        base,_=objective(model,data);pair,_=endpoint_agreement(model,data)
        total=base+pair;assert torch.isfinite(total)
        total.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        assert torch.isfinite(norm);optimizer.step()
        if update%32==0:evaluate(update)
    result=dict(time=time.time(),seconds=time.time()-began,source=commit,device='CPU',updates=128,
        objective='unchanged v56 objective + 1.0 * negative log inner-product of same-task successful endpoint distributions',
        fit_episodes=fit,heldout_training_episodes=heldout,history=history,
        no_hand_assigned_symbol_ids=True,no_nonterminal_failure_labels=True,no_runtime_change=True,
        original_dev80_used=False,finalfixed80_used=False,new_rollouts=0,live_training_modified=False,
        limitation='Heldout is disjoint only for this additional fitting. Source model used the broader training corpus. This does not measure navigation or prove bisimulation.')
    torch.save(dict(model=model.state_dict(),config=model.export_config(),source=commit),out/'probe.pt')
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('fit_episodes','heldout_training_episodes','history')}),flush=True)

if __name__=='__main__':main()
