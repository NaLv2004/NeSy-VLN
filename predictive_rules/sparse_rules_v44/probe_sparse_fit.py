"""Fixed-training-data comparison: dense and learned sparse successor support.

The live v43 training continues. Candidate models never execute navigation.
The split is episode-disjoint for this decoder-fitting diagnostic only; the
source model has previously trained on the broader training corpus.
"""
from collections import defaultdict
import importlib.util,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import FactoredWorld,Config,hard
from sparse_distribution import loss as sparse_loss
from data import ROOT,sha

def load_dense():
    path=ROOT/'src_boundary_balanced_v43/model.py'
    spec=importlib.util.spec_from_file_location('v43_dense_reference',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module.FactoredWorld

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.20)
    torch.backends.cuda.matmul.allow_tf32=True;torch.manual_seed(441017);began=time.time()
    out=ROOT/'diagnostic_sparse_support_v44';out.mkdir(exist_ok=True)
    assert not (out/'report.json').exists(),'This immutable diagnostic already completed'
    if not (out/'source.pt').exists():
        live=ROOT/'run_boundary_balanced_v43';commit=json.loads((live/'best.COMMITTED.json').read_text())
        os.link(live/commit['filename'],out/'source.pt');(out/'source.COMMITTED.json').write_text(json.dumps(commit))
    commit=json.loads((out/'source.COMMITTED.json').read_text());assert sha(out/'source.pt')==commit['sha256']
    saved=torch.load(out/'source.pt',map_location='cpu',weights_only=False)
    dense_class=load_dense();encoder=dense_class(Config(**saved['model_config'])).eval().cuda()
    encoder.load_state_dict(saved['model']);c=encoder.cfg
    groups=defaultdict(list);records={}
    for path in sorted((ROOT/'onpolicy_successors_v40/counterfactual_onpolicy').glob('*.json')):
        record=json.loads(path.read_text());assert record['training_only'] and not record['goal_supervision']
        groups[(record['domain'],record['category'],record['distance_group'])].append(path);records[path]=record
    rng=random.Random(441017);holdout=set()
    for group in groups.values():holdout.update(rng.sample(group,2))
    arrays=defaultdict(list);episode_names=[]
    with torch.inference_mode():
        for path,record in records.items():
            with np.load(path.with_suffix('.npz')) as archive:
                x=torch.from_numpy(archive['features'].copy())[None].cuda()
                recorded=archive['recorded_actions'];n=x.shape[1]
                positions=sorted(set([0,min(1,n-1),min(5,n-1),n-1]))
                successor=torch.from_numpy(archive['successor_features'][positions].copy()).cuda()
            previous=torch.tensor([[17]+recorded[:n-1].tolist()],device='cuda')
            logits,z,_=encoder.observation_encoding(x);ob=hard(logits)
            lm,_,_=encoder.memory_sequence(ob,previous,z);mem=hard(lm)
            context=encoder.context_sequence(x,ob,previous,torch.tensor([0],device='cuda'))
            slogits,sz,_=encoder.observation_encoding(successor)
            current_m=mem[0,positions,None].expand(-1,17,-1);current_o=ob[0,positions,None].expand(-1,17,-1)
            actions=torch.arange(17,device='cuda')[None].expand(len(positions),-1)
            target_m,_=encoder.memory_step(current_m,hard(slogits),actions,sz)
            for i,t in enumerate(positions):
                arrays['context'].append(context[0,max(0,t-5)].cpu().numpy())
                arrays['memory'].append(current_m[i].cpu().numpy());arrays['observation'].append(current_o[i].cpu().numpy())
                arrays['target_m'].append(target_m[i].argmax(-1).cpu().numpy())
                arrays['target_o'].append(slogits[i].argmax(-1).cpu().numpy())
                arrays['heldout'].append(path in holdout);arrays['t'].append(t);episode_names.append(record['episode'])
    del encoder;torch.cuda.empty_cache()
    data={key:torch.tensor(np.asarray(value),device='cuda') for key,value in arrays.items()}
    np.savez_compressed(out/'fixed_training_cases.npz',**{k:v.cpu().numpy() for k,v in data.items()})
    fit=(~data['heldout']).nonzero()[:,0];test=data['heldout'].nonzero()[:,0]
    descriptor=dict(training_only=True,source_commit=commit,fit_episodes=sorted({episode_names[i] for i in fit.tolist()}),
        heldout_training_episodes=sorted({episode_names[i] for i in test.tolist()}),
        fit_positions=len(fit),heldout_positions=len(test),context_age='min(t,5)',seed=441017,
        label_encoders_frozen=True,holdout_scope='Decoder updates in this diagnostic only; source model previously saw these training episodes')
    assert not set(descriptor['fit_episodes'])&set(descriptor['heldout_training_episodes'])
    (out/'dataset.json').write_text(json.dumps(descriptor,indent=2))
    keys=('definition_generator.','observation_embedding.','action_embedding.','rule.','persistence.','action_persistence.','memory_rule.','memory_predictor.')
    results={}
    for variant,cls in [('dense',dense_class),('entmax15',FactoredWorld)]:
        torch.manual_seed(441017);model=cls(Config(**saved['model_config'])).eval().cuda();model.load_state_dict(saved['model'])
        model.requires_grad_(False);parameters=[p for name,p in model.named_parameters() if name.startswith(keys)]
        for parameter in parameters:parameter.requires_grad_(True)
        optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=1e-4)
        def predict(indices):
            arguments=(data['context'][indices],data['memory'][indices],data['observation'][indices],torch.arange(17,device='cuda')[None].expand(len(indices),-1))
            if variant=='entmax15':return model.transition(*arguments,return_scores=True)
            m,o=model.transition(*arguments);return m,o,o
        @torch.no_grad()
        def measure(indices):
            totals=defaultdict(float);count=0
            for batch in indices.split(64):
                pm,logp,_=predict(batch);p=logp.exp();truth=data['target_o'][batch];tm=data['target_m'][batch]
                actual=p.gather(-1,truth[...,None])[...,0];onehot=F.one_hot(truth,64).float()
                totals['observation_correct']+=int((p.argmax(-1)==truth).sum())
                totals['memory_correct']+=int((pm.argmax(-1)==tm).sum())
                totals['excluded_true']+=int((actual==0).sum());totals['support']+=int((p>0).sum())
                totals['brier_sum']+=float((p-onehot).square().sum())
                totals['capped_nll_sum']+=float(-actual.clamp_min(1e-12).log().sum());count+=truth.numel()
            return dict(outcomes=count,observation_accuracy=totals['observation_correct']/count,
                memory_accuracy=totals['memory_correct']/count,true_outcome_excluded_rate=totals['excluded_true']/count,
                mean_support=totals['support']/count,brier=totals['brier_sum']/count,nll_capped_at_1e_12=totals['capped_nll_sum']/count)
        history=[];tick=time.time()
        for update in range(801):
            if update%200==0:
                row=dict(update=update,fit=measure(fit),heldout_train_episodes=measure(test),seconds=time.time()-tick)
                history.append(row);print(json.dumps(dict(variant=variant,**row)),flush=True)
            if update==800:break
            indices=fit[torch.randint(len(fit),(64,),device='cuda')]
            pm,logp,scores=predict(indices);tm=data['target_m'][indices];to=data['target_o'][indices]
            objective=F.cross_entropy(pm.flatten(0,1),tm.flatten())
            objective=objective+(sparse_loss(scores,to) if variant=='entmax15' else F.nll_loss(logp.flatten(0,1),to.flatten()))
            optimizer.zero_grad(set_to_none=True);objective.backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,5.);assert torch.isfinite(objective) and torch.isfinite(norm)
            optimizer.step()
        results[variant]=history
        torch.save(dict(model=model.state_dict(),source_commit=commit,diagnostic_only=True),out/f'{variant}_decoder.pt')
        del model,optimizer;torch.cuda.empty_cache()
    report=dict(dataset=descriptor,updates_per_candidate=800,results=results,seconds=time.time()-began,
        cuda_peak_gib=torch.cuda.max_memory_allocated()/2**30,live_training_changed=False,
        limitation='Frozen encoder, training-cache decoder probe; not physical navigation or independent heldout-layout success')
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(complete=True,seconds=report['seconds'],cuda_peak_gib=report['cuda_peak_gib'])),flush=True)
if __name__=='__main__':main()
