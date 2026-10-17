"""Transfer the full v11 model and distill its rules into a definition-bound head."""
import json,os,time,subprocess
from pathlib import Path
import torch
from predictive_nesy.visual_memory_subgoal_model import VisualMemorySubgoalModel,Config
from predictive_nesy.definition_conditioned_world import DefinitionConditionedWorld
from predictive_nesy.subgoal_data import load_windows,batch
from predictive_nesy.data import sha
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
out=root/'initialization_definition_world_v12';out.mkdir(exist_ok=True)
assert not (out/'initial.pt').exists()
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert {int(x.split(',')[0]):int(x.split(',')[1]) for x in raw.splitlines()}[7]<300
torch.set_num_threads(2);torch.manual_seed(1964)
source=root/'run_vocabulary_rules_v11/checkpoint_003200.pt'
saved=torch.load(source,map_location='cpu',weights_only=False)
assert sha(source)==json.loads(source.with_suffix('.COMMITTED.json').read_text())['sha256']
teacher=VisualMemorySubgoalModel(Config(**saved['model_config'])).cuda().eval();teacher.load_state_dict(saved['model']);teacher.requires_grad_(False)
student=DefinitionConditionedWorld(Config(**saved['model_config'])).cuda().eval()
state=student.state_dict();copied=[]
for n,v in saved['model'].items():
    if not n.startswith('transitions.'):
        assert n in state and state[n].shape==v.shape
        state[n]=v;copied.append(n)
student.load_state_dict(state)
for n,p in student.named_parameters():p.requires_grad_(n.startswith('definition_rules.'))
fixed={n:v.cpu().clone() for n,v in student.state_dict().items() if not n.startswith('definition_rules.')}
windows,provenance=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
contexts=[];defs=[]
with torch.no_grad():
    for start in sorted({w['start'] for w in windows}):
        selected=[w for w in windows if w['start']==start]
        for offset in range(0,len(selected),64):
            d=batch(selected[offset:offset+64],'cuda');width=min(start+1,11)
            x=teacher.context_read(teacher.embed(d['history'][:,:width]))+teacher.action_embed(d['previous_actions'][:,:width])
            h,_=teacher.history(x,teacher.task(d['task'])[None]);c=teacher.context_norm(h[torch.arange(len(x),device='cuda'),d['history_length']-1])
            contexts.append(c);defs.append(teacher.definitions(c))
contexts=torch.cat(contexts);defs=torch.cat(defs)
opt=torch.optim.AdamW(student.definition_rules.parameters(),lr=1e-3,weight_decay=1e-4);began=time.time()
for step in range(1,1601):
    ids=torch.randint(len(contexts),(64,),device='cuda');context=contexts[ids];definitions=defs[ids]
    with torch.no_grad():target=teacher.transitions(context).float().softmax(-1)
    predicted=student.definition_rules(definitions,context)
    kl=(target*(target.clamp_min(1e-8).log()-predicted.clamp_min(1e-8).log())).sum(-1).mean()
    opt.zero_grad();kl.backward();norm=torch.nn.utils.clip_grad_norm_(student.definition_rules.parameters(),1.);assert torch.isfinite(norm);opt.step()
    if step==1 or step%200==0:
        row=dict(update=step,kl=float(kl.detach()),seconds=time.time()-began)
        with (out/'distillation.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
with torch.no_grad():
    ids=torch.arange(0,len(contexts),max(1,len(contexts)//128),device='cuda')[:128]
    match=0;count=0;divergence=0
    for offset in range(0,len(ids),32):
        ix=ids[offset:offset+32];target=teacher.transitions(contexts[ix]).float().softmax(-1);predicted=student.definition_rules(defs[ix],contexts[ix])
        match+=int((target.argmax(-1)==predicted.argmax(-1)).sum());count+=target.shape[0]*target.shape[1]*target.shape[2]
        divergence+=float((target*(target.clamp_min(1e-8).log()-predicted.clamp_min(1e-8).log())).sum(-1).sum())
    d=defs[:2];c=contexts[:2];perm=torch.randperm(student.cfg.states,device='cuda')
    original=student.definition_rules(d,c);renamed=student.definition_rules(d[:,perm],c)
    permutation_error=float((renamed-original[:,perm][:,:,:,perm]).abs().max())
    assert permutation_error<1e-6
assert all(torch.equal(v,student.state_dict()[n].cpu()) for n,v in fixed.items())
manifest=dict(saved['manifest'],architecture='anonymous_definition_conditioned_world_v12',
    definition_rule_initialization=dict(source=str(source),source_sha256=sha(source),updates=1600,
        teacher_rule_agreement=match/count,teacher_rule_kl=divergence/count,permutation_error=permutation_error,
        copied_non_rule_tensors=len(copied),semantic_priors=False),data=provenance)
path=out/'initial.pt';torch.save(dict(model=student.cpu().state_dict(),model_config=student.export_config(),manifest=manifest,update=0),path)
commit=dict(path=path.name,sha256=sha(path),update=0,time=time.time());path.with_suffix('.COMMITTED.json').write_text(json.dumps(commit,indent=2))
report=dict(complete=True,checkpoint=commit,all_non_rule_weights_unchanged=True,teacher_rule_agreement=match/count,
    teacher_rule_kl=divergence/count,state_rename_equivariance_error=permutation_error,
    rule_parameters_generated_from_learned_definitions=True,manual_state_semantics=False,seconds=time.time()-began,
    navigation_improvement_proven=False)
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
