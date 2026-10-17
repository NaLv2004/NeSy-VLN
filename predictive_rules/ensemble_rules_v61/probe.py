"""Train-only calibration pilot; never selects a navigation action.

All categorical encoders, goal definitions and memory dynamics are frozen at
v57 best1000. Three observation-rule heads receive independently bootstrapped
episodes; a matched single-head control sees the full fitting partition.
The24 held-out episodes supply no gradient in this pilot. They are training
layouts and are not claimed to be unseen in the source model's training.
"""
import copy,json,os,random,select,sys,time,traceback
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
sys.path.insert(0,str(ROOT/'src_onpolicy_refresh_v60_training'))
from model import FactoredWorld,Config,hard
from data import sha
DATA=ROOT/'onpolicy_successors_v60';OUT=ROOT/'diagnostic_ensemble_rules_v61'
def read(p):return json.loads(p.read_text())
def write(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2);os.nice(10);torch.manual_seed(611017)
    OUT.mkdir(exist_ok=True);assert not (OUT/'result.json').exists()
    start=time.time();collector=read(DATA/'launcher.json')['pid']
    p=Path(f'/proc/{collector}/cmdline')
    if p.exists() and p.read_bytes():
        assert b'coordinate_collection.py' in p.read_bytes()
        fd=os.pidfd_open(collector)
        write(OUT/'status.json',dict(state='waiting_collection',pid=os.getpid(),time=time.time(),collector_pid=collector))
        try:select.select([fd],[],[])
        finally:os.close(fd)
    assert read(DATA/'status.json')['state']=='complete'
    manifest=read(DATA/'manifest.json');assert manifest['training_only']
    commit=read(DATA/'source.COMMITTED.json');assert sha(DATA/'source.pt')==commit['sha256']
    saved=torch.load(DATA/'source.pt',map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).eval().requires_grad_(False)
    model.load_state_dict(saved['model'],strict=True);del saved
    groups=defaultdict(list);metas={}
    for path in sorted((DATA/'counterfactual_onpolicy').glob('*.json')):
        r=read(path);assert r['training_only'] and r['scene'] in manifest['train_layouts']
        metas[r['episode']]=(path,r);groups[(r['domain'],r['distance_group'],r['category'])].append(r['episode'])
    split_rng=random.Random(611017);held=set()
    for key,group in sorted(groups.items()):
        group=list(group);split_rng.shuffle(group);held.update(group[:2])
    assert len(held)==24 and len(metas)==120
    episode_rows=defaultdict(list);raw=[];observed_matches=[];recorded_predictions=[]
    write(OUT/'status.json',dict(state='encoding_fixed_targets',pid=os.getpid(),time=time.time()))
    with torch.inference_mode():
        for episode,(path,r) in metas.items():
            npz=path.with_suffix('.npz');assert sha(npz)==r['output_sha256']
            with np.load(npz,allow_pickle=False) as a:
                x=torch.tensor(a['features'].copy())[None]
                successors=torch.tensor(a['successor_features'].copy())
                actions=torch.tensor(a['recorded_actions'].copy()).long()
            n=x.shape[1];previous=torch.cat([torch.tensor([17]),actions])[:n][None]
            task=torch.tensor([['chair','table'].index(r['category'])])
            contexts=model.context_sequence(x,hard(model.observe(x)),previous,task)
            z=F.normalize(model.visual(x.float().flatten(-2)),dim=-1)
            future_z=F.normalize(model.visual(successors.float().flatten(-2)),dim=-1)
            ae=model.action_embedding(torch.arange(17)).clone()
            preserve=(model.operator_class(torch.arange(17)).argmax(-1)==1).float()
            for j,program in enumerate(r['programs']):
                boundary=program['start'];end=r['programs'][j+1]['start'] if j+1<len(r['programs']) else n
                context=contexts[:,boundary]
                defs=F.normalize(model.observation_definitions[None]+model.observation_definition_delta(context),dim=-1)[0]
                mdefs=model.definitions(context)[0]
                labels=(future_z@defs.T).argmax(-1)
                current_o=(z[0]@defs.T).argmax(-1)
                # For an unexecutable last position, the failed proof still
                # records the exact available integer state. No fake action.
                for t in range(boundary,end):
                    state=r['steps'][t] if t<len(r['steps']) else r['failed_proof']
                    assert state is not None and 'memory' in state and 'observed' in state
                    m,o=int(state['memory']),int(state['observed'])
                    observed_matches.append(int(current_o[t])==o)
                    row=dict(memory=mdefs[m].clone(),observation=F.one_hot(torch.tensor(o),64).float(),
                        definitions=defs.clone(),context=context[0].clone(),target=labels[t].clone(),
                        current=o,episode=episode)
                    episode_rows[episode].append(len(raw));raw.append(row)
    # Clone outside inference mode so these tensors may participate in backward.
    tensors={k:torch.stack([r[k] for r in raw]).clone() for k in ('memory','observation','definitions','context','target')}
    ae=ae.clone();preserve=preserve.clone()
    idx={part:torch.tensor([i for e,indices in episode_rows.items() if (e in held)==(part=='holdout') for i in indices]) for part in ('fit','holdout')}
    def batch(indices):
        d={k:v[indices] for k,v in tensors.items()};b=len(indices)
        return (d['memory'][:,None].expand(-1,17,-1),d['observation'][:,None].expand(-1,17,-1),d['definitions'],
                ae[None].expand(b,-1,-1),d['context'],preserve[None,:,None].expand(b,-1,-1)),d['target'],d['observation'].argmax(-1)
    fit_names=[e for e in sorted(episode_rows) if e not in held]
    heads=[copy.deepcopy(model.definition_rule).requires_grad_(True) for _ in range(4)]
    optimizers=[torch.optim.AdamW(h.parameters(),lr=1e-4,weight_decay=1e-4) for h in heads]
    samplers=[random.Random(611017+i) for i in range(4)]
    pools=[idx['fit'].tolist()]
    bootstrap=[]
    for k in range(1,4):
        names=samplers[k].choices(fit_names,k=len(fit_names));bootstrap.append(names)
        pools.append([i for e in names for i in episode_rows[e]])
    def statistics(prob,target,current):
        predicted=prob.argmax(-1);hit=predicted==target;changed=target!=current[:,None]
        actual=prob.gather(-1,target[...,None]).squeeze(-1)
        conf=prob.max(-1).values
        ece=0.
        for low in np.arange(0,1,.1):
            mask=(conf>=low)&(conf<low+.100001)
            if mask.any():ece+=float(mask.float().mean()*(conf[mask].mean()-hit[mask].float().mean()).abs())
        return dict(n_positions=len(target),accuracy=float(hit.float().mean()),
            changed_accuracy=float(hit[changed].float().mean()) if changed.any() else None,
            nll=float(-actual.clamp_min(1e-9).log().mean()),mean_confidence=float(conf.mean()),
            actual_outcome_probability=float(actual.mean()),brier=float((prob-F.one_hot(target,64)).square().sum(-1).mean()),ece=ece)
    @torch.no_grad()
    def evaluate(update):
        reports={}
        for part,indices in idx.items():
            ps=[[] for _ in range(5)];ts=[];cs=[]
            for block in indices.split(64):
                inputs,target,current=batch(block)
                for i,head in enumerate([model.definition_rule]+heads):ps[i].append(head(*inputs).exp())
                ts.append(target);cs.append(current)
            p=[torch.cat(items) for items in ps];target=torch.cat(ts);current=torch.cat(cs)
            ensemble=torch.stack(p[2:]).mean(0)
            reports[part]=dict(baseline=statistics(p[0],target,current),single_control=statistics(p[1],target,current),
                members=[statistics(q,target,current) for q in p[2:]],ensemble=statistics(ensemble,target,current),
                member_argmax_disagreement=float((torch.stack(p[2:]).argmax(-1)!=p[2].argmax(-1)).any(0).float().mean()))
        return dict(update=update,results=reports,seconds=time.time()-start)
    history=[evaluate(0)]
    for update in range(1,401):
        for i,(head,opt) in enumerate(zip(heads,optimizers)):
            chosen=torch.tensor(samplers[i].choices(pools[i],k=64));inputs,target,_=batch(chosen)
            logp=head(*inputs);loss=F.nll_loss(logp.flatten(0,1),target.flatten())
            opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),5.)
            assert torch.isfinite(loss) and torch.isfinite(norm);opt.step()
        if update%100==0:
            row=evaluate(update);history.append(row)
            write(OUT/'status.json',dict(state='fitting_heads',pid=os.getpid(),time=time.time(),**row))
            print(json.dumps(row),flush=True)
    result=dict(time=time.time(),seconds=time.time()-start,source=commit,updates=400,threads=2,device='CPU',
        fit_episodes=fit_names,holdout_episodes=sorted(held),bootstrap_episodes=bootstrap,
        source_observed_id_reproduction=float(np.mean(observed_matches)),history=history,
        navigation_evaluations=0,official_dev80_used=False,finalfixed80_used=False,
        neural_planning_calls=0,formal_training_unchanged=True,
        limitation='Partial bootstrap heads share frozen perception/context/memory and source initialization. This is not a full independently initialized Deep Ensembles reproduction; no formal robust-control guarantee. Held-out pilot episodes belong to training layouts and may have been seen by the source model.')
    torch.save(dict(heads=[h.state_dict() for h in heads],source=commit,model_config=model.export_config()),OUT/'heads.pt')
    write(OUT/'result.json',result);write(OUT/'status.json',dict(state='complete',pid=os.getpid(),time=time.time(),updates=400))
    print(json.dumps(dict(state='complete',seconds=result['seconds'],heldout=history[-1]['results']['holdout'])),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        OUT.mkdir(exist_ok=True);write(OUT/'status.json',dict(state='failed',pid=os.getpid(),time=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
