import torch
import torch.nn.functional as F
from .generator import program_distance


def advantages(policy,gamma=.99,lam=.95):
    rows=[r for c in policy.chunks for r in c['rows']]
    carry=0.;next_value=0.
    for row in reversed(rows):
        delta=row['reward']+gamma*next_value-row['value']
        carry=delta+gamma*lam*carry
        row['advantage']=carry;row['return']=carry+row['value'];next_value=row['value']


def train_chunk(model,chunk,expert,normalization=(0.,1.),program=None,tokens=None,next_tokens=None):
    cfg=model.cfg; weights=cfg['loss']; device=model.device
    if program is None:program=model.generator(chunk['task'],chunk['images'],chunk['feedback'],chunk['previous'])
    memory=list(chunk['memory']);prior=None; terms={};observations=[];ratios=[]
    def add(name,x):terms[name]=terms.get(name,0.)+x
    for index,row in enumerate(chunk['rows']):
        result=model.step(program,row['image'],memory,prior,tokens=None if tokens is None else tokens[index:index+1])
        dist=torch.distributions.Categorical(probs=result['probs'])
        a=row['action']; action=torch.tensor(a,device=device)
        logp=dist.log_prob(action)
        if expert:add('bc',-logp)
        else:
            ratio=(logp-row['old_logp']).exp()
            advantage=(row['advantage']-normalization[0])/normalization[1]
            add('ppo',-torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage))
            add('value',F.smooth_l1_loss(result['value'],torch.tensor(row['return'],device=device)))
            ratios.append(ratio.detach())
        add('entropy',-dist.entropy())
        _,outcome=model.grounder.readout(program.definitions,result['belief'],result['confidence'],a)
        if not expert:
            labels=torch.tensor([float(row['collision']),row['feedback'][1],float(row['success'])],device=device)
            add('outcome',F.binary_cross_entropy_with_logits(outcome[[0,2]],labels[[0,2]])+
                F.mse_loss(outcome[1].sigmoid(),labels[1]))
        elif a==0:add('outcome',F.binary_cross_entropy_with_logits(outcome[2],outcome.new_tensor(1.)))
        prior,memory,conflict=model.advance(program,result,a,row['feedback'],memory)
        target,confidence=model.transition_target(program,row['next_image'],memory,
            tokens=None if next_tokens is None else next_tokens[index:index+1])
        # Observability weights come from a stopped-gradient EMA teacher; floor prevents loss masking.
        mask=confidence.detach().clamp_min(.2)
        add('transition',(F.binary_cross_entropy(prior[0].clamp(1e-5,1-1e-5),target.detach(),reduction='none')*mask).mean())
        add('conflict',conflict)
        # Confidence cannot zero out all evidence. No semantic predicate target is supplied.
        add('diversity',(result['observed_confidence'].mean()-.7).square())
        observations.append(result['observation'])
    n=max(1,len(chunk['rows']));terms={k:v/n for k,v in terms.items()}
    regular=model.logic.regularization(program)
    terms['sparse']=regular['selector_entropy']+.1*regular['rule_weight']
    definitions=F.normalize(program.definitions,dim=-1)
    gram=definitions@definitions.T
    diversity=((gram-torch.eye(len(gram),device=device))**2).mean()
    if len(observations)>1:
        diversity=diversity+(.02-torch.stack(observations).std(0,unbiased=False)).clamp_min(0).mean()
    terms['diversity']=terms.get('diversity',0)+diversity
    loss=sum(weights.get(k,1.)*v for k,v in terms.items())
    stats={k:float(v.detach()) for k,v in terms.items()}
    if ratios:stats['ppo_ratio_mean']=float(torch.stack(ratios).mean())
    return loss,stats


def train_batch(model,batch,normalization=(0.,1.)):
    """Actual batched VLM programs and all frame encodings, with recurrent per-episode logic."""
    from .runtime import image_tensor
    requests=[(c['task'],c['images'],c['feedback'],c['previous']) for c,_ in batch]
    programs=model.generator.generate_many(requests)
    rows=[row for c,_ in batch for row in c['rows']]
    images=torch.cat([image_tensor(row['image'],model.cfg['image_size'],model.device) for row in rows])
    tokens=model.grounder.encode(images)
    with torch.no_grad():
        next_images=torch.cat([image_tensor(row['next_image'],model.cfg['image_size'],model.device) for row in rows])
        next_tokens=model.teacher.encode(next_images)
    offset=0;losses=[];statistics=[]
    for (chunk,expert),program in zip(batch,programs):
        n=len(chunk['rows'])
        loss,stats=train_chunk(model,chunk,expert,normalization,program,tokens[offset:offset+n],next_tokens[offset:offset+n])
        losses.append(loss);statistics.append(stats);offset+=n
    return torch.stack(losses).mean(),statistics


def train_revision(model,trial):
    c=trial['chunk']
    predicted=model.generator(c['task'],c['images'],c['feedback'],c['previous'])
    return program_distance(predicted,trial['target'])*model.cfg['loss']['revision']
