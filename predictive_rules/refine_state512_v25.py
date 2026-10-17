"""Distill training-only visual clusters into complete neural state definitions."""
import json,os,re,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from predictive_nesy.evidence_rule_world import EvidenceRuleWorld,Config
from predictive_nesy.state512_rule_world import State512RuleWorld
from predictive_nesy.subgoal_data import load_windows,batch
from predictive_nesy.raw_boundary_data import RawBoundaryFeatures
from predictive_nesy.data import sha

def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    torch.set_num_threads(2);torch.manual_seed(2525);torch.backends.cuda.matmul.allow_tf32=True
    root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'initialization_state512_v25r1'
    if out.exists():assert not any(out.iterdir()),'Existing initialization evidence must not be overwritten'
    else:out.mkdir()
    began=time.time()
    source=root/'run_evidence_rule_v23/checkpoint_001000.pt';assert sha(source)==json.loads(source.with_suffix('.COMMITTED.json').read_text())['sha256']
    saved=torch.load(source,map_location='cpu',weights_only=False)
    dictionary_path=root/'diagnostics/state_resolution_v24_centers_512.pt'
    dictionary=torch.load(dictionary_path,map_location='cpu',weights_only=False)
    assert dictionary['not_deployed'] and dictionary['source_sha256']==sha(source)
    centers=F.normalize(dictionary['centers'].float(),dim=-1).cuda();assert centers.shape==(512,256)
    fitting=set(dictionary['fitting_layouts']);excluded=set(dictionary['excluded_training_layouts'])
    assert not fitting&excluded and {f'layout_{x}' for x in fitting|excluded}==set(saved['manifest']['data']['train_layouts'])
    old=EvidenceRuleWorld(Config(**saved['model_config'])).cuda().eval();old.load_state_dict(saved['model']);old.requires_grad_(False)
    model=State512RuleWorld(Config(states=512)).cuda().eval();state=model.state_dict();copied=[]
    for name,value in saved['model'].items():
        if name in state and state[name].shape==value.shape:state[name]=value;copied.append(name)
    model.load_state_dict(state)
    windows,provenance=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
    source_rows=json.loads((Path(saved['manifest']['data']['source'])/'training_rows.json').read_text())
    episode_layout={row['episode_id']:int(re.search(r'P25_train_(\d+)_',row.get('source_episode_id',row['episode_id'])).group(1)) for row in source_rows}
    contexts=[];ordered=[]
    with torch.no_grad():
        for start in sorted({w['start'] for w in windows}):
            selected=[w for w in windows if w['start']==start]
            for begin in range(0,len(selected),64):
                subset=selected[begin:begin+64];d=batch(subset,'cuda');width=min(start+1,11)
                x=old.context_read(old.embed(d['history'][:,:width]))+old.action_embed(d['previous_actions'][:,:width])
                hidden,_=old.history(x.float(),old.task(d['task']).float()[None])
                raw=hidden[torch.arange(len(subset),device='cuda'),d['history_length']-1]
                contexts.append(old.context_norm(raw));ordered.extend(subset)
    contexts=torch.cat(contexts);fit=torch.tensor([i for i,w in enumerate(ordered) if episode_layout[w['episode_id']] in fitting],device='cuda')
    held=torch.tensor([i for i,w in enumerate(ordered) if episode_layout[w['episode_id']] in excluded],device='cuda')
    assert len(fit) and len(held)
    with torch.no_grad():
        original_centers=F.normalize(old.definitions(contexts[fit[:128]]).mean(0),dim=-1)
        parents=(centers@original_centers.T).argmax(-1)
        model.definitions.queries[0].weight.copy_(old.definitions.queries[0].weight[parents])
        model.subgoal.weight.copy_(old.subgoal.weight[parents])
        counts=torch.bincount(parents,minlength=128).clamp_min(1)
        model.subgoal.bias.copy_(old.subgoal.bias[parents]-counts[parents].float().log())
    candidate=root/'initialization_state512_v25/unaccepted_candidate.pt'
    assert sha(candidate)==json.loads(candidate.with_suffix('.COMMITTED.json').read_text())['sha256']
    resumed=torch.load(candidate,map_location='cpu',weights_only=False)
    model.load_state_dict(resumed['model'])
    del old,state,resumed
    for name,param in model.named_parameters():param.requires_grad_(name.startswith('definitions.'))
    fixed={n:v.cpu().clone() for n,v in model.state_dict().items() if not n.startswith('definitions.')}
    optimizer=torch.optim.AdamW(model.definitions.parameters(),lr=5e-5,weight_decay=0.,eps=1e-10)
    for step in range(1,12001):
        lr=5e-5 if step<=3000 else 1e-5
        for group in optimizer.param_groups:group['lr']=lr
        ids=fit[torch.randint(len(fit),(16,),device='cuda')]
        predicted=model.definitions(contexts[ids]);loss=256*F.mse_loss(predicted,centers[None].expand_as(predicted))
        optimizer.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.definitions.parameters(),1.)
        assert torch.isfinite(norm);optimizer.step()
        if step==1 or step%1000==0:
            record=dict(state='distilling',pid=os.getpid(),step=step,mse=float(loss.detach())/256,lr=lr,seconds=time.time()-began,time=time.time())
            atomic(out/'status.json',record)
            with (out/'distillation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
    assert all(torch.equal(v,model.state_dict()[n].cpu()) for n,v in fixed.items())
    features=RawBoundaryFeatures(root/'prequant_visual_features_v18','cuda');model.eval();diagnostics={}
    with torch.no_grad():
        for label,indices in [('fit_contexts',fit),('excluded_training_contexts',held)]:
            cosines=[]
            for ids in indices.split(32):cosines.append((F.normalize(model.definitions(contexts[ids]),dim=-1)*centers).sum(-1).cpu())
            values=torch.cat(cosines);diagnostics[label]=dict(contexts=len(indices),mean_cosine=float(values.mean()),minimum_cosine=float(values.min()))
        agreements=0;observations=0;endpoint_alias=0;moving=0
        for start in sorted({w['start'] for w in windows}):
            selected=[w for w in windows if w['start']==start]
            for begin in range(0,len(selected),32):
                subset=selected[begin:begin+32];d=batch(subset,'cuda');width=min(start+1,11);raw,available=features.demonstrations(subset)
                p=model.prepare(d['task'],d['history'][:,:width],d['previous_actions'][:,:width],d['history_length'],start=start,boundary_features=raw)
                observed=model.training_targets(p,d['targets'],d['actions']).argmax(-1).squeeze(-1)
                target=model.training_targets(dict(p,definitions=centers[None].expand(len(subset),-1,-1)),d['targets'],d['actions']).argmax(-1).squeeze(-1)
                mask=d['state_valid'];agreements+=int(((observed==target)&mask).sum());observations+=int(mask.sum())
                for i,w in enumerate(subset):
                    if w['actions'][0]!=0:moving+=1;endpoint_alias+=int(observed[i,0]==observed[i,int(w['goal_index'])])
    passed=diagnostics['excluded_training_contexts']['mean_cosine']>.999 and agreements/observations>.975
    manifest={**saved['manifest'],'architecture':'state512_source_evidence_world_v25','data':provenance,
        'model':model.export_config(),'learned_joint_state_count':512,
        'definition_initialization':dict(source_checkpoint=str(source),source_sha256=sha(source),clustering_file_sha256=sha(dictionary_path),
            fitted_training_layouts=sorted(fitting),excluded_training_layouts=sorted(excluded),definitions_distillation_steps=24000,
            refinement_initialization_sha256=sha(candidate),refinement_loss_scaling=256,refinement_weight_decay=0.,
            semantic_labels_used=False,actions_used_for_clustering=False,full_neural_definition_decoder=True,runtime_cluster_file_required=False),
        'state_refinement_initialization':dict(nearest_pretrained_visual_parent=parents.cpu().tolist(),
            scope='Only initializes embedding queries and classifier weights by visual similarity; no parent IDs exported at runtime'),
        'initialization_copied_tensors':copied}
    path=out/('initial.pt' if passed else 'unaccepted_candidate.pt');tmp=path.with_suffix('.tmp')
    torch.save(dict(model=model.cpu().state_dict(),model_config=model.export_config(),manifest=manifest,update=0),tmp);os.replace(tmp,path)
    commit=dict(path=path.name,sha256=sha(path),update=0,time=time.time());atomic(path.with_suffix('.COMMITTED.json'),commit)
    report=dict(passed=passed,checkpoint=commit,diagnostics=diagnostics,non_definition_weights_unchanged=True,
        cluster_assignment_agreements=agreements,observations=observations,moving_endpoint_alias=endpoint_alias,moving_windows=moving,
        semantic_priors=False,runtime_cluster_file_required=False,seconds=time.time()-began,time=time.time())
    atomic(out/'report.json',report);atomic(out/'status.json',dict(state='complete' if passed else 'rejected',pid=os.getpid(),time=time.time(),report=report))
    print(json.dumps(report),flush=True)
    if not passed:raise RuntimeError('State decoder was preserved without acceptance; inspect evidence before continuing')

if __name__=='__main__':main()
