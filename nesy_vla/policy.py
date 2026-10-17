import copy
import torch
from .runtime import pil,public_feedback
from .logic import NoSupportedAction


class EpisodePolicy:
    def __init__(self,model,category,revision=True):
        self.model=model; self.task='Find the '+category+' and stop close to it.'
        self.program=None; self.memory=[]; self.prior=None; self.images=[]; self.feedback=[]
        self.steps=0; self.revision=revision; self.chunk=None; self.chunks=[]
        self.logic_snapshot=({n:v.detach().cpu().clone() for n,v in model.logic.state_dict().items()}
            if model.cfg.get('discrete_reasoning',False) else None)

    def prepare(self,image):
        self.images.append(image)
        boundary=self.program is None or (self.revision and self.steps%self.model.cfg['revision_interval']==0)
        if boundary:
            previous=None if self.program is None else self.program.detach()
            self.chunk={'task':self.task,'images':list(self.images),'feedback':copy.deepcopy(self.feedback),
                'previous':previous,'memory':[x.detach() for x in self.memory],'rows':[],
                'start':self.steps}
            return self.task,self.images,self.feedback,previous
        return None

    def install(self,program):
        self.program=program.detach();self.program.validate();self.prior=None
        self.chunk['program']=self.program;self.chunks.append(self.chunk)

    def choose(self,image,stochastic=True,action=None,prepared=False,tokens=None):
        if not prepared:
            request=self.prepare(image)
            if request is not None:self.install(self.model.generator(*request))
        result=self.model.step(self.program,image,self.memory,self.prior) if tokens is None else self.model.step(
            self.program,image,self.memory,self.prior,tokens=tokens)
        dist=torch.distributions.Categorical(probs=result['probs'])
        selected=int(dist.sample()) if stochastic else int(result['probs'].argmax())
        if action is not None:selected=int(action)
        row={'image':image,'action':selected,'old_logp':float(dist.log_prob(torch.tensor(selected,device=self.model.device))),
            'value':float(result['value']),'belief':result['belief'].detach().cpu().tolist(),
            'confidence':result['confidence'].detach().cpu().tolist(),'support':result['support'].detach().cpu().tolist(),
            'contribution':result['action_contribution'].detach().cpu().tolist(),'version':self.program.version}
        row['observed_truth']=result['observation'].detach().cpu().tolist()
        if 'symbol_codes' in result:
            row['symbol_codes']=result['symbol_codes'].detach().cpu().tolist()
            row['rule_gates']=result['rule_gates'].detach().cpu().tolist()
            row['symbol_state_id']=''.join(map(str,row['symbol_codes']))
            row['perceptual_codes']=result['perceptual_codes'].detach().cpu().tolist()
            row['prior_codes']=None if result['prior_codes'] is None else result['prior_codes'].detach().cpu().tolist()
        row['state_inference']=[{k:v.detach().cpu().tolist() for k,v in layer.items()} for layer in result['layers']]
        return selected,result,row

    def observe(self,row,result,next_image,response,expert=False,next_tokens=None):
        feedback,vector=public_feedback(response,row['image'],next_image)
        prior,memory,conflict=self.model.advance(self.program,result,row['action'],vector,self.memory)
        self.prior=tuple(x.detach() for x in prior);self.memory=[x.detach() for x in memory]
        with torch.no_grad():
            target,_=self.model.transition_target(self.program,next_image,self.memory,tokens=next_tokens)
        prediction_error=float((prior[0]-target).abs().mean())
        rounded=lambda x:[round(float(v),2) for v in x.detach().cpu()]
        self.feedback.append({'action':row['action'],**feedback,'prediction_error':round(prediction_error,4),
            'predicate_state':rounded(result['belief']),'confidence':rounded(result['confidence']),
            'state_prediction_error':rounded(prior[0]-target)})
        if 'symbol_codes' in row:self.feedback[-1]['symbol_codes']=list(row['symbol_codes'])
        row['predicted_next_truth']=prior[0].detach().cpu().tolist()
        row['observed_next_truth']=target.detach().cpu().tolist()
        row.update(next_image=next_image,feedback=vector,collision=feedback['collision'],
            prediction_error=prediction_error,reward=float(response.get('reward',0.)),
            success=bool(response.get('metrics',{}).get('success',False)),done=bool(response.get('done',False)),
            expert=expert)
        if 'training_reward_audit' in response:row['training_reward_audit']=response['training_reward_audit']
        self.chunk['rows'].append(row); self.steps+=1


@torch.no_grad()
def rollout(model,sim,episode,stochastic=True,revision=True,reward_overrides=None):
    from .reward_experiment import training_reward
    response=sim.call('reset',episode=episode); image=pil(response['rgb'])
    policy=EpisodePolicy(model,episode['category'],revision)
    failed=None
    for step in range(model.cfg['horizon']):
        try:action,result,row=policy.choose(image,stochastic)
        except NoSupportedAction as e:failed=str(e);break
        response=training_reward(sim.call('step',action=action),reward_overrides); after=pil(response['rgb'])
        policy.observe(row,result,after,response);image=after
        if response['done']:break
    metrics=copy.deepcopy(response.get('metrics',{}))
    if failed:metrics.update(success=False,inference_failure=failed)
    metrics['success']=bool(metrics.get('success',False))
    return policy,metrics


@torch.no_grad()
def rollout_many(model,simulators,episodes,stochastic=True,revision=True,reward_overrides=None):
    from .reward_experiment import training_reward
    from concurrent.futures import ThreadPoolExecutor
    from .runtime import image_tensor
    policies=[EpisodePolicy(model,e['category'],revision) for e in episodes]
    metrics=[None]*len(episodes)
    with ThreadPoolExecutor(max_workers=len(episodes)) as pool:
        responses=list(pool.map(lambda x:x[0].call('reset',episode=x[1]),zip(simulators,episodes)))
        images=[pil(r['rgb']) for r in responses];active=list(range(len(episodes)))
        for step in range(model.cfg['horizon']):
            requests=[];indices=[]
            for i in active:
                request=policies[i].prepare(images[i])
                if request is not None:requests.append(request);indices.append(i)
            if requests:
                programs=model.generator.generate_many(requests)
                for i,program in zip(indices,programs):policies[i].install(program)
            tensors=torch.cat([image_tensor(images[i],model.cfg['image_size'],model.device) for i in active])
            tokens=model.grounder.encode(tensors);selected=[];valid=[]
            for offset,i in enumerate(active):
                try:
                    selected.append(policies[i].choose(images[i],stochastic,prepared=True,tokens=tokens[offset:offset+1]));valid.append(i)
                except NoSupportedAction as e:metrics[i]={'success':False,'inference_failure':str(e)}
            replies=list(pool.map(lambda item:simulators[item[0]].call('step',action=item[1][0]),zip(valid,selected)))
            replies=[training_reward(response,reward_overrides) for response in replies]
            afters=[pil(response['rgb']) for response in replies]
            teacher_tokens=model.teacher.encode(torch.cat([image_tensor(im,model.cfg['image_size'],model.device) for im in afters])) if afters else None
            following=[]
            for offset,(i,(action,result,row),response,after) in enumerate(zip(valid,selected,replies,afters)):
                policies[i].observe(row,result,after,response,next_tokens=teacher_tokens[offset:offset+1]);images[i]=after
                if response['done']:metrics[i]=copy.deepcopy(response['metrics'])
                else:following.append(i)
            active=following
            if not active:break
    for i in range(len(metrics)):
        if metrics[i] is None:metrics[i]={'success':False,'horizon':model.cfg['horizon']}
    return policies,metrics


@torch.no_grad()
def demonstration(model,bank,episode_id):
    from PIL import Image
    rows=bank.demos[episode_id][:model.cfg['horizon']]
    policy=EpisodePolicy(model,rows[0]['category'])
    policy.episode_id=episode_id
    images=[Image.open(bank.demo_image_path(x)).convert('RGB') for x in rows]
    for i,(row,image) in enumerate(zip(rows,images)):
        action,result,record=policy.choose(image,action=row['action'])
        record['supervision_mask']=row.get('supervision_mask',True)
        if 'execution_outcome_labels' in row:record['execution_outcome_labels']=row['execution_outcome_labels']
        after=images[min(i+1,len(images)-1)]
        response={'feedback':row.get('public_feedback',{}),'reward':0.,'done':i==len(rows)-1,
                  'metrics':{'success':i==len(rows)-1 and action==0}}
        policy.observe(record,result,after,response,expert=True)
    return policy


@torch.no_grad()
def demonstrations_many(model,bank,episode_ids):
    """Replay successful supervision with batched VLM and image encoders, separate beliefs."""
    from PIL import Image
    from .runtime import image_tensor
    rows=[bank.demos[e][:model.cfg['horizon']] for e in episode_ids]
    images=[[Image.open(bank.demo_image_path(r)).convert('RGB') for r in episode] for episode in rows]
    policies=[EpisodePolicy(model,r[0]['category']) for r in rows]
    for policy,episode_id in zip(policies,episode_ids):policy.episode_id=episode_id
    for step in range(max(map(len,rows))):
        active=[i for i,r in enumerate(rows) if step<len(r)]
        requests=[];indices=[]
        for i in active:
            request=policies[i].prepare(images[i][step])
            if request is not None:requests.append(request);indices.append(i)
        if requests:
            for i,program in zip(indices,model.generator.generate_many(requests)):policies[i].install(program)
        current=model.grounder.encode(torch.cat([image_tensor(images[i][step],model.cfg['image_size'],model.device) for i in active]))
        afters=[images[i][min(step+1,len(images[i])-1)] for i in active]
        following=model.teacher.encode(torch.cat([image_tensor(im,model.cfg['image_size'],model.device) for im in afters]))
        for offset,i in enumerate(active):
            action,result,record=policies[i].choose(images[i][step],stochastic=False,action=rows[i][step]['action'],
                prepared=True,tokens=current[offset:offset+1])
            record['supervision_mask']=rows[i][step].get('supervision_mask',True)
            if 'execution_outcome_labels' in rows[i][step]:
                record['execution_outcome_labels']=rows[i][step]['execution_outcome_labels']
            response={'feedback':rows[i][step].get('public_feedback',{}),'reward':0.,'done':step==len(rows[i])-1,
                      'metrics':{'success':step==len(rows[i])-1 and action==0}}
            policies[i].observe(record,result,afters[offset],response,expert=True,next_tokens=following[offset:offset+1])
    return policies
