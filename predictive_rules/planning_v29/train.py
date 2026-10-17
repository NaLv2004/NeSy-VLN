"""One isolated experiment: arbitrary-boundary rules plus budgeted renewal.

Renewal supervision compares old/fresh programs on recorded training actions.
The GRU minimizes expected proof/action and dynamics cost plus a generation
penalty. Its input is only hard observation/prediction/goal IDs and past action
IDs. The development set is never used for gradient updates or gate targets.
"""
import argparse
import copy
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from planning_model import PlanningObservedWorld as AdaptiveObservedWorld, Config, representation_loss, branch_representation_loss, perception_proof_loss
from predictive_nesy.data import read, sha
from predictive_nesy.subgoal_data import batch, TransitionBank
from predictive_nesy.replayed_boundary_data import ReplayedBoundaryFeatures
from predictive_nesy.selected_row_objectives import frozen_rule_proof_levels
from predictive_nesy.graph_subgoal_model import straight_through
from evaluate_state512_v25 import atomic

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')


class Windows:
    def __init__(self, source, layouts):
        source=Path(source);self.rows=read(source/'training_rows.json');self.codes=np.load(source/'visual_tokens.npy')
        assert read(source/'manifest.json')['data_provenance']['train_layouts']==sorted(layouts)
        self.groups=defaultdict(list);self.keys=[]
        for i,row in enumerate(self.rows):
            match=re.search(r'P25_train_(\d+)_',row.get('source_episode_id',row['episode_id']))
            assert match and 'layout_'+match.group(1) in layouts
            self.groups[row['episode_id']].append(i)
        for name,indices in self.groups.items():
            assert self.rows[indices[-1]]['label']==0
            self.keys.extend((name,t) for t in range(min(32,len(indices))))
        self.by_start={t:[k for k in self.keys if k[1]==t] for t in range(32)}

    def window(self,name,start):
        indices=self.groups[name];n=len(indices);width=min(start+1,11);begin=start-width+1
        history=np.zeros((11,49),np.int64);history[:width]=self.codes[indices[begin:start+1]]
        previous=np.full(11,17,np.int64)
        previous[:width]=[self.rows[indices[t-1]]['label'] if t else 17 for t in range(begin,start+1)]
        h=min(10,32-start,n-start)
        targets=self.codes[[indices[min(start+i,n-1)] for i in range(11)]].astype(np.int64)
        actions=np.zeros(10,np.int64);actions[:h]=[self.rows[indices[start+i]]['label'] for i in range(h)]
        supervised=np.zeros(10,bool);supervised[:h]=[self.rows[indices[start+i]].get('supervised',True) for i in range(h)]
        return dict(episode_id=name,task=['chair','table'].index(self.rows[indices[0]]['category']),start=start,
                    history=history,previous_actions=previous,history_length=width,targets=targets,actions=actions,
                    action_valid=np.arange(10)<h,state_valid=np.arange(11)<=h,supervised=supervised,goal_index=h)


def prepare(model, data, features, rows):
    raw,available=features.demonstrations(rows)
    width=int(data['history_length'][0]);start=int(data['start'][0])
    return model.prepare(data['task'],data['history'][:,:width],data['previous_actions'][:,:width],
                         data['history_length'],start=start,boundary_features=raw,boundary_available=available)


@torch.no_grad()
def action_cost(model,p,predicted,observed,action,next_state,horizon,table=None):
    if table is None:table=model.rows(p)
    state=model.correction.selected(p['definitions'],p['context_raw'],predicted,observed).argmax(-1)
    index=torch.arange(len(state),device=state.device)
    dynamics=-table[index,state,action,next_state].clamp_min(1e-8).log()
    goal=F.one_hot(p['subgoal_logits'].argmax(-1).flatten(),512).float()
    levels=frozen_rule_proof_levels(table,goal,horizon)
    scores=levels[horizon][index,state]/(10000*(horizon+1)*.5)
    imitation=F.cross_entropy(scores,action,reduction='none').clamp_max(12)
    return dynamics.clamp_max(12)+imitation


def observed_objective(model,data,p,return_parts=False,target_labels=None):
    """Train the actual hard feedback recurrence, including correction rules.

    Each recorded future frame is converted to one hard observation ID by the
    online encoder; successor targets use the EMA encoder. Only categorical
    IDs reach correction/dynamics during execution.
    Forward state propagation is categorical; straight-through derivatives
    train its tables without replacing execution by a soft belief.
    """
    b=len(data['task']);device=data['task'].device;h=min(10,32-int(data['start'][0]))
    with torch.no_grad():
        observations=model.training_targets(p,data['targets'],data['actions']).argmax(-1).squeeze(-1)
        labels=observations if target_labels is None else target_labels
    full=model.rows(p)
    target_goal=labels[torch.arange(b,device=device),data['goal_index']]
    goal_logits=p['subgoal_logits'].squeeze(1)
    goal_loss=F.cross_entropy(goal_logits,target_goal)
    levels=frozen_rule_proof_levels(full.detach(),straight_through(goal_logits.softmax(-1)),h)
    state=observations[:,0];losses=[];proof_losses=[];corrections=[];correct=[];matches=[];end_states=[]
    index=torch.arange(b,device=device)
    for t in range(h):
        logits=model.correction.selected(p['definitions'],p['context_raw'],state,observations[:,t])
        # Both tables use the encoder's anonymous state vocabulary. Learn C
        # from those observed IDs, not from an action-fitting dynamics loss
        # that can relabel the present to an unrelated convenient state.
        current=F.one_hot(logits.argmax(-1),512).float()
        rows=full[index[:,None],torch.arange(512,device=device)[None],data['actions'][:,t,None]]
        successor=torch.einsum('bs,bsk->bk',current,rows)
        # The selected hard premise is constant to the dynamics gradient.
        row_target=rows.gather(-1,labels[:,t+1,None,None].expand(-1,512,1)).squeeze(-1)
        losses.append(-(current*row_target.clamp_min(1e-8).log()).sum(-1))
        corrections.append(F.cross_entropy(logits,observations[:,t],reduction='none'))
        # Runtime replans from each newly observed state to the retained goal.
        # Its proof budget is ten, limited only by the 32-action episode budget.
        depth=min(10,32-int(data['start'][0])-t)
        # Proof imitation trains the generated goal. It must not encourage
        # correction to invent a convenient source state to fit an action.
        scores=torch.einsum('bs,bsa->ba',current.detach(),levels[depth])/(10000*(h+1)*.5)
        proof_losses.append(F.cross_entropy(scores,data['actions'][:,t],reduction='none'))
        matches.append(scores.argmax(-1)==data['actions'][:,t])
        state=successor.argmax(-1);correct.append(state==labels[:,t+1]);end_states.append(state)
    mask=data['action_valid'][:,:h].float();supervised=mask*data['supervised'][:,:h].float()
    def average(values,weights):return (torch.stack(values,1)*weights).sum()/weights.sum().clamp_min(1)
    dynamics=average(losses,mask);correction=average(corrections,mask);proof=average(proof_losses,supervised)
    loss=2*dynamics+goal_loss+proof+correction
    endpoints=torch.stack(end_states,1)[index,data['goal_index'].clamp_max(h)-1]
    output=(loss,dict(loss=float(loss.detach()),observed_conditioned_dynamics=float(dynamics.detach()),
        correction_loss=float(correction.detach()),proof_action_imitation=float(proof.detach()),subgoal_prediction=float(goal_loss.detach()),
        transition_accuracy=float(average([v.float() for v in correct],mask)),
        proof_action_accuracy=float(average([v.float() for v in matches],supervised)),
        subgoal_exact_accuracy=float((goal_logits.argmax(-1)==target_goal).float().mean()),
        teacher_path_exact_goal=float((endpoints==target_goal).float().mean())),labels)
    if return_parts:return (*output,dict(dynamics=dynamics,proof=proof,correction=correction,goal=goal_loss))
    return output


def save(path,model,optimizer,update,manifest,dual,rng,teacher):
    temporary=path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),model_config=model.export_config(),optimizer=optimizer.state_dict(),ema_teacher=teacher.state_dict(),
        update=update,manifest=manifest,renewal_dual=dual,python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),temporary)
    os.replace(temporary,path);atomic(path.with_suffix('.COMMITTED.json'),dict(sha256=sha(path),update=update,time=time.time()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--updates',type=int,default=2000);parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--eval-every',type=int,default=1000);args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    out=Path(args.out);assert out.resolve().is_relative_to(ROOT) and not out.exists();out.mkdir(parents=True)
    rng=random.Random(26161);torch.manual_seed(26161);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    source=Path(args.source);assert sha(source)==read(source.with_suffix('.COMMITTED.json'))['sha256']
    old=torch.load(source,map_location='cuda',weights_only=False)
    model=AdaptiveObservedWorld(Config(**old['model_config'])).cuda()
    missing,unexpected=model.load_state_dict(old['model'],strict=False)
    assert not unexpected and all(k.startswith(('revision.','correction.','inverse_discrete.')) for k in missing)
    representation_prefixes=('tokens.','observation.','definitions.','visual_memory.','memory_projection.',
                             'context_read.','history.','context_norm.','task.','action_embed.')
    parameters=('definition_rules.','subgoal.','raw_','goal_','rule_evidence.','revision.','correction.','inverse_discrete.')+representation_prefixes
    for name,p in model.named_parameters():p.requires_grad_(name.startswith(parameters))
    frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
    corpus=Windows(old['manifest']['data']['source'],old['manifest']['data']['train_layouts'])
    features=ReplayedBoundaryFeatures(ROOT/'prequant_visual_features_v18',ROOT/'replayed_boundary_features_v20','cuda')
    banks=[TransitionBank(x['path'],old['manifest']['data']['train_layouts']) for x in old['manifest']['transition_banks']]
    manifest=dict(**{k:old['manifest'][k] for k in ['data','frozen_vision']},architecture='planning_grounded_observation_v29',
        initialize=str(source),initialize_sha256=sha(source),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        observation_ids=512,transition='argmax correction[predicted,observed] followed by argmax action[state,action]',
        planner='exact backward derivation to the unchanged generated goal; no neural actor fallback',
        renewal='GRU consuming hard IDs and past actions; expected old/fresh proof cost plus generation penalty',
        renewal_target_probability=.12,renewal_minimum_generation_penalty=.2,
        correction_gradient_contract='C supervised by observed anonymous IDs only; dynamics and proof cannot relabel the present',
        renewal_cost_contract='Exact retained C/T recurrence and retained definition labels; fresh C/T evaluated on the same teacher action',
        renewal_budget_control='Observed hard gate firing rate; cost-sensitive BCE plus expected generation cost',
        no_fixed_renewal_interval=True,actual_five_step_mean_interval_not_yet_verified=True,
        learned_semantic_state_labels=False,initial_perception_frozen=False,frontend_CNN_frozen=True,
        representation_training='Hard backward proof action supervision reaches perception through the actual categorical C table; balanced all-action dynamics and rule inverse; EMA successor targets',
        perception_proof_weight=.5,demonstration_inverse_weight=.1,branch_batch_size=32,
        branch_inverse_equivalence='Exactly equal recorded patch codes receive a shared action target set; no semantic labels',
        inverse_input_contract='Two hard one-hot categorical states only; never images or image features',
        inverse_used_at_runtime=False,ema_decay=.99,
        arbitrary_training_boundary_windows=len(corpus.keys),training_args=vars(args),created=time.time(),
        transition_banks=[dict(path=str(x.directory),**x.refresh()) for x in banks])
    atomic(out/'manifest.json',manifest)
    optimizer=torch.optim.AdamW([
        dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(representation_prefixes)],lr=1e-5),
        dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith(representation_prefixes)],lr=1e-4)],weight_decay=1e-4)
    teacher=copy.deepcopy(model).eval().requires_grad_(False)
    del old;dual=1.;began=time.time()
    for update in range(1,args.updates+1):
        t=time.time();model.train();start=rng.choice(corpus.keys)[1]
        keys=rng.choices(corpus.by_start[start],k=args.batch_size);rows=[corpus.window(*key) for key in keys]
        data=batch(rows,'cuda');p=prepare(model,data,features,rows)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_program=prepare(teacher,data,features,rows)
            targets=teacher.training_targets(teacher_program,data['targets'],data['actions']).argmax(-1).squeeze(-1)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss,metrics,labels=observed_objective(model,data,p,target_labels=targets)
            representation,extra=representation_loss(model,p,data,targets)
            loss=loss+representation;metrics.update(extra)
            perception_proof,proof_metrics=perception_proof_loss(model,p,data)
            loss=loss+.5*perception_proof;metrics.update(proof_metrics)
        with torch.no_grad():labels=model.training_targets(p,data['targets'],data['actions']).argmax(-1).squeeze(-1)
        observed=labels[:,0]
        predicted=torch.randint(512,observed.shape,device='cuda')
        correction=F.cross_entropy(model.correction.selected(p['definitions'],p['context_raw'],predicted,observed),observed)
        loss=loss+correction
        gate_loss=loss.new_zeros(());probability=0.;benefit=0.
        if start:
            age=rng.randint(1,start);old_start=start-age
            old_rows=[corpus.window(name,old_start) for name,_ in keys]
            old_data=batch(old_rows,'cuda')
            with torch.no_grad():
                old_p=prepare(model,old_data,features,old_rows)
                prefix=torch.tensor(np.asarray([corpus.codes[corpus.groups[name][old_start:start+1]] for name,_ in keys]),device='cuda').long()
                symbols=model.encode_sequence(old_p,prefix)[0].argmax(-1)
                old_probabilities=model.rows(old_p);old_table=old_probabilities.argmax(-1)
                previous_actions=torch.tensor([[corpus.rows[corpus.groups[name][q-1]]['label'] if q else 17 for q in range(old_start,start+1)] for name,_ in keys],device='cuda')
                projected=symbols.clone()
                for q in range(1,symbols.shape[1]):
                    corrected=model.correction.selected(old_p['definitions'],old_p['context_raw'],projected[:,q-1],symbols[:,q-1]).argmax(-1)
                    projected[:,q]=old_table[torch.arange(len(keys),device='cuda'),corrected,previous_actions[:,q]]
                next_codes=torch.tensor(np.asarray([corpus.codes[corpus.groups[name][min(start+1,len(corpus.groups[name])-1)]] for name,_ in keys]),device='cuda').long()
                old_next=model.encode_sequence(old_p,torch.cat([prefix,next_codes[:,None]],1))[0][:,-1].argmax(-1)
                old_cost=action_cost(model,old_p,projected[:,-1],symbols[:,-1],data['actions'][:,0],old_next,min(10,32-start),old_probabilities)
                fresh_cost=action_cost(model,p,observed,observed,data['actions'][:,0],labels[:,1],min(10,32-start))
                ages=torch.arange(age+1,device='cuda')[None].expand(len(keys),-1)
                lengths=torch.full((len(keys),),age+1,device='cuda',dtype=torch.long)
                goal=old_p['subgoal_logits'].argmax(-1).flatten()
            logits=model.revision.sequence(symbols,projected,previous_actions,data['task'],ages,lengths,goal=goal)
            prob=logits.sigmoid()
            # Costs are stopped-gradient measurements of old/fresh rule quality.
            # Renewal gradients cannot secretly become an action-selection head.
            advantage=(old_cost-fresh_cost).clamp(-12,12)
            net_benefit=advantage-dual-.2
            target=(net_benefit>0).float()
            cost_sensitive=F.binary_cross_entropy_with_logits(logits,target,reduction='none')*net_benefit.abs().clamp(.2,4.)
            gate_loss=(-prob*net_benefit+.25*cost_sensitive).mean()
            loss=loss+.1*gate_loss
            probability=float(prob.detach().mean());benefit=float(advantage.mean())
            hard_rate=float((logits.detach()>=0).float().mean())
            dual=max(0.,min(50.,dual+.05*(hard_rate-.12)))
            metrics['renewal_hard_rate']=hard_rate
            metrics['renewal_cost_optimal_target_rate']=float(target.mean())
        if not torch.isfinite(loss):raise FloatingPointError(metrics)
        total_loss=float(loss.detach());loss.backward()
        # Protect all 17 action branches, including actions absent from the
        # successful demonstration. Only encoded successor images supervise T.
        selected=rng.choices(banks,weights=[len(x.keys) for x in banks],k=1)[0]
        branches=selected.sample(rng,min(32,args.batch_size),'cuda')
        raw,available=features.transitions(branches,selected.directory)
        width=branches['history'].shape[1]
        lengths=torch.full((len(branches['task']),),width,device='cuda',dtype=torch.long)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            branch_program=model.prepare(branches['task'],branches['history'],branches['previous_actions'],lengths,
                start=branches['start'],boundary_features=raw,boundary_available=available)
            with torch.no_grad():
                teacher_branch=teacher.prepare(branches['task'],branches['history'],branches['previous_actions'],lengths,
                    start=branches['start'],boundary_features=raw,boundary_available=available)
            branch_loss,branch_metrics=branch_representation_loss(model,teacher,branch_program,teacher_branch,branches)
        if not torch.isfinite(branch_loss):raise FloatingPointError(branch_metrics)
        branch_loss.backward();total_loss+=float(branch_loss.detach());metrics.update(branch_metrics)
        groups=dict(world=('definition_rules.','raw_','rule_evidence.'),goal=('subgoal.','goal_'),
                    correction=('correction.',),revision=('revision.',),definitions=('definitions.',),
                    perception=('tokens.','observation.','visual_memory.','memory_projection.'),
                    context=('context_read.','history.','context_norm.','task.','action_embed.'),inverse=('inverse_discrete.',))
        for group,prefixes in groups.items():
            norm=torch.nn.utils.clip_grad_norm_([p for n,p in model.named_parameters() if p.requires_grad and n.startswith(prefixes)],1.)
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite '+group+' gradients')
            metrics[group+'_gradient_norm']=float(norm)
        optimizer.step()
        with torch.no_grad():
            online=model.state_dict()
            for name,value in teacher.state_dict().items():
                if value.is_floating_point():value.lerp_(online[name],.01)
                else:value.copy_(online[name])
        torch.cuda.synchronize()
        record=dict(**metrics,update=update,time=time.time(),seconds=time.time()-t,total_loss=total_loss,
                    random_prior_correction_loss=float(correction.detach()),renewal_loss=float(gate_loss.detach()),
                    expected_renewal_probability=probability,renewal_dual=dual,renewal_measured_cost_gain=benefit,
                    memory_peak_gib=torch.cuda.max_memory_allocated()/2**30)
        with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        atomic(out/'status.json',dict(state='training',pid=os.getpid(),elapsed_seconds=time.time()-began,**record))
        if update==1 or update%50==0:print(json.dumps(record),flush=True)
        if update==1 or update%100==0 or update==args.updates:save(out/'latest.pt',model,optimizer,update,manifest,dual,rng,teacher)
        if update%args.eval_every==0:
            checkpoint=out/f'checkpoint_{update:06d}.pt';save(checkpoint,model,optimizer,update,manifest,dual,rng,teacher)
            torch.cuda.empty_cache()
            evaluation=out/f'development_step_{update:06d}'
            with (out/f'evaluation_{update:06d}.log').open('wb') as log:
                rc=subprocess.call([sys.executable,'-u',str(Path(__file__).with_name('evaluate.py')),'--checkpoint',str(checkpoint),'--out',str(evaluation)],stdout=log,stderr=subprocess.STDOUT)
            if rc:atomic(out/'evaluation_error.json',dict(update=update,returncode=rc))
            else:
                result=read(evaluation/'summary.json')
                with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(dict(update=update,**result))+'\n')
                print(json.dumps(dict(stage='development_result',update=update,**result)),flush=True)
    assert all(torch.equal(value,dict(model.named_parameters())[name].detach().cpu()) for name,value in frozen.items())
    atomic(out/'status.json',dict(state='training_budget_complete',pid=os.getpid(),update=args.updates,time=time.time(),goal_complete=False))


if __name__=='__main__':
    try:main()
    except Exception as error:
        import traceback
        if '--out' in sys.argv:
            folder=Path(sys.argv[sys.argv.index('--out')+1])
            if folder.exists():atomic(folder/'status.json',dict(state='failed',pid=os.getpid(),time=time.time(),
                error=repr(error),traceback=traceback.format_exc(),goal_complete=False))
        raise
