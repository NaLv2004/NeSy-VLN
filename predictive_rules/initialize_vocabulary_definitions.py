"""Distill automatically learned visual centers into the full neural decoder.

The deployed model still generates all definition vectors with its neural
decoder. The clustering file is a training target and is never loaded by the
controller or evaluator. No semantic labels enter this initialization.
"""
import hashlib,json,os,random,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config
from predictive_nesy.subgoal_data import load_windows,batch
from predictive_nesy.data import sha
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
out=root/'initialization_vocabulary_v11r1';out.mkdir(exist_ok=True)
assert not (out/'initial.pt').exists(),'Existing initialization must be inspected rather than overwritten'
torch.set_num_threads(2);torch.manual_seed(1842)
source=root/'run_grounded_aggregate_v10/checkpoint_001200.pt'
saved=torch.load(source,map_location='cpu',weights_only=False)
assert sha(source)==json.loads(source.with_suffix('.COMMITTED.json').read_text())['sha256']
dictionary_path=root/'diagnostics/unsupervised_vocabulary_v9/centers.pt'
dictionary=torch.load(dictionary_path,map_location='cpu',weights_only=False)
centers=F.normalize(dictionary['centers'].float(),dim=-1).cuda();fitting=set(dictionary['fit_episode_ids'])
# Transitions, subgoal, and definition decoder are deliberately fresh. Their
# old index semantics do not match this independently learned vocabulary.
model=VisualMemorySubgoalModel(Config(**saved['model_config'])).cuda()
state=model.state_dict();copied=[]
for name,value in saved['model'].items():
    if not name.startswith(('definitions.','transitions.','subgoal.')):
        state[name]=value;copied.append(name)
model.load_state_dict(state);model.eval()
for name,param in model.named_parameters():param.requires_grad_(name.startswith('definitions.'))
fixed={name:value.detach().cpu().clone() for name,value in model.state_dict().items() if not name.startswith('definitions.')}
windows,provenance=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
contexts=[];context_fit=[];context_windows=[]
with torch.no_grad():
    for start in sorted({w['start'] for w in windows}):
        selected=[w for w in windows if w['start']==start]
        for offset in range(0,len(selected),64):
            subset=selected[offset:offset+64];d=batch(subset,'cuda');width=min(start+1,11)
            x=model.context_read(model.embed(d['history'][:,:width]))+model.action_embed(d['previous_actions'][:,:width])
            states,_=model.history(x.float(),model.task(d['task']).float()[None])
            raw=states[torch.arange(len(subset),device='cuda'),d['history_length']-1]
            contexts.append(model.context_norm(raw));context_fit.extend(w['episode_id'] in fitting for w in subset);context_windows.extend(subset)
contexts=torch.cat(contexts);fit_ids=torch.tensor([i for i,v in enumerate(context_fit) if v],device='cuda')
held_ids=torch.tensor([i for i,v in enumerate(context_fit) if not v],device='cuda')
optimizer=torch.optim.AdamW(model.definitions.parameters(),lr=1e-3,weight_decay=1e-4);began=time.time()
for step in range(1,12001):
    lr=1e-3 if step<=600 else 1e-4 if step<=3000 else 2e-5
    for group in optimizer.param_groups:group['lr']=lr
    ids=fit_ids[torch.randint(len(fit_ids),(64,),device='cuda')]
    predicted=model.definitions(contexts[ids]);loss=F.mse_loss(predicted,centers[None].expand_as(predicted))
    optimizer.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.definitions.parameters(),1.)
    assert torch.isfinite(norm);optimizer.step()
    if step==1 or step%1000==0:
        record=dict(step=step,mse=float(loss.detach()),lr=lr,seconds=time.time()-began)
        with (out/'distillation.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
assert all(torch.equal(value,model.state_dict()[name].detach().cpu()) for name,value in fixed.items())
with torch.no_grad():
    diagnostics={}
    for name,indices in [('fit',fit_ids),('held_out_training',held_ids)]:
        all_cos=[]
        for offset in range(0,len(indices),64):
            definitions=F.normalize(model.definitions(contexts[indices[offset:offset+64]]),dim=-1)
            all_cos.append((definitions*centers).sum(-1).cpu())
        cosine=torch.cat(all_cos);diagnostics[name]=dict(contexts=len(indices),mean_center_cosine=float(cosine.mean()),minimum_center_cosine=float(cosine.min()))
    alias=0;total_moving=0;cluster_agreement=0;observations=0
    for start in sorted({w['start'] for w in windows}):
        selected=[w for w in windows if w['start']==start]
        for offset in range(0,len(selected),32):
            subset=selected[offset:offset+32];d=batch(subset,'cuda');width=min(start+1,11)
            p=model.generate(d['task'],d['history'][:,:width],d['previous_actions'][:,:width],d['history_length'],start=start)
            labels=model.training_targets(p,d['targets'],d['actions']).argmax(-1).squeeze(-1)
            oracle={**p,'definitions':centers[None].expand(len(subset),-1,-1)}
            target=model.training_targets(oracle,d['targets'],d['actions']).argmax(-1).squeeze(-1)
            mask=d['state_valid'];cluster_agreement+=int(((labels==target)&mask).sum());observations+=int(mask.sum())
            for i,w in enumerate(subset):
                if w['actions'][0]!=0:
                    total_moving+=1;alias+=int(labels[i,0]==labels[i,int(w['goal_index'])])
    diagnostic=dict(diagnostics=diagnostics,moving_endpoint_alias=alias,moving_windows=total_moving,
        cluster_assignment_agreements=cluster_agreement,observations=observations)
    (out/'candidate_diagnostics.json').write_text(json.dumps(diagnostic,indent=2));print(json.dumps(diagnostic),flush=True)
    if diagnostics['held_out_training']['mean_center_cosine']<=.99 or cluster_agreement/observations<=.97 or alias>=30:
        torch.save(dict(model=model.cpu().state_dict(),optimizer=optimizer.state_dict(),diagnostic=diagnostic),out/'unaccepted_candidate.pt')
        raise RuntimeError('Neural decoder does not yet reproduce the learned vocabulary; candidate preserved without promotion')
manifest={**saved['manifest'],
    'initialization_kind':'neural definition decoder distilled from unsupervised training-only visual clusters',
    'definition_initialization':dict(source_checkpoint=str(source),source_sha256=sha(source),clustering_file_sha256=sha(dictionary_path),
        clustering_fit_episodes=len(fitting),semantic_labels_used=False,actions_used_for_clustering=False,
        full_neural_definition_decoder=True,runtime_cluster_file_required=False,definitions_distillation_steps=12000),
    'data':provenance,'initialization_copied_tensors':copied}
payload=dict(model=model.cpu().state_dict(),model_config=model.export_config(),manifest=manifest,update=0)
path=out/'initial.pt';temporary=path.with_suffix('.tmp');torch.save(payload,temporary);os.replace(temporary,path)
commit=dict(path=path.name,sha256=sha(path),update=0,time=time.time());path.with_suffix('.COMMITTED.json').write_text(json.dumps(commit,indent=2))
result=dict(passed=True,checkpoint=commit,diagnostics=diagnostics,all_non_definition_weights_unchanged=True,
    copied_representation_tensors=len(copied),fresh_rule_and_goal_heads=True,
    moving_endpoint_alias=alias,moving_windows=total_moving,cluster_assignment_agreements=cluster_agreement,
    observations=observations,seconds=time.time()-began,semantic_priors=False,runtime_cluster_file_required=False)
(out/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
