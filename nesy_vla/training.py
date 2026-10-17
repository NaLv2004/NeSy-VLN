import torch
import torch.nn.functional as F
from .generator import program_distance
from .losses import transition_loss,inverse_transition_logits


def advantages(policy,gamma=.99,lam=.95):
    rows=[r for c in policy.chunks for r in c['rows']]
    carry=0.;next_value=0.
    for row in reversed(rows):
        delta=row['reward']+gamma*next_value-row['value']
        carry=delta+gamma*lam*carry
        row['advantage']=carry;row['return']=carry+row['value'];next_value=row['value']


def train_chunk(model,chunk,expert,normalization=(0.,1.),program=None,tokens=None,next_tokens=None,return_loss_terms=False,reference_tokens=None):
    cfg=model.cfg; weights=cfg['loss']; device=model.device
    if program is None:program=model.generator(chunk['task'],chunk['images'],chunk['feedback'],chunk['previous'])
    memory=list(chunk['memory']);prior=None; terms={};observations=[];ratios=[];inverse_correct=0;inverse_count=0
    expert_correct=expert_count=0;bit_correct=bit_total=copy_correct=changed_correct=changed_total=0;symbol_mismatch=0;symbol_total=0
    def add(name,x):terms[name]=terms.get(name,0.)+x
    for index,row in enumerate(chunk['rows']):
        replay_codes=row['perceptual_codes'] if not expert and cfg.get('discrete_reasoning',False) else None
        result=model.step(program,row['image'],memory,prior,tokens=None if tokens is None else tokens[index:index+1],
            replay_perceptual_codes=replay_codes)
        if replay_codes is not None:
            expected=torch.tensor(replay_codes,device=device,dtype=torch.long)
            add('on_policy_fresh_bit_disagreement_metric',(result['fresh_perceptual_codes']!=expected).float().mean())
        if reference_tokens is not None:
            if not expert:raise ValueError('Photometric augmentation cannot alter an on-policy PPO replay')
            with torch.no_grad():
                reference,_=model.grounder.observe(reference_tokens[index:index+1],program.definitions.detach(),
                    [value.detach() for value in memory])
            add('visual_consistency',F.mse_loss(result['observation'],reference))
            add('quantization',(result['observation']-(result['observation'].detach()>=.5).float()).square().mean())
            add('view_bit_disagreement_metric',((result['observation'].detach()>=.5)!=(reference>=.5)).float().mean())
        dist=torch.distributions.Categorical(probs=result['probs'])
        a=row['action']; action=torch.tensor(a,device=device)
        logp=dist.log_prob(action)
        if expert:
            supervised=row.get('supervision_mask',True)
            if type(supervised) is not bool:raise TypeError('Supervision mask must be Boolean')
            if supervised and cfg.get('counterfactual_symbol_bc',False):
                from .counterfactual_learning import counterfactual_symbol_bc
                from .symbols import DiscreteState
                state=DiscreteState(result['symbol_codes'],result['symbol_values'])
                add('bc',counterfactual_symbol_bc(model.logic,program,state,a,
                    probability_floor=torch.finfo(result['probs'].dtype).eps))
            elif supervised:add('bc',-logp)
            if supervised:
                expert_count+=1;expert_correct+=int(result['probs'].detach().argmax()==a)
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
        if expert and cfg.get('execution_outcome_supervision',False) and 'execution_outcome_labels' in row:
            from .symbols import DiscreteState
            from .outcome_supervision import execution_loss
            if weights.get('execution',0)<=0:raise ValueError('Execution supervision needs an explicit positive loss weight')
            state=DiscreteState(result['symbol_codes'][None],result['symbol_values'][None])
            definitions=program.definitions
            if cfg.get('detach_execution_symbols',False):
                state=DiscreteState(state.codes,state.values.detach());definitions=definitions.detach()
            logits=model.execution_decoder(definitions,state)
            labels=logits.new_tensor(row['execution_outcome_labels'])[None]
            positive_weight=logits.new_tensor(cfg.get('execution_positive_weight',[1.,1.]))
            add('execution',execution_loss(logits,labels,positive_weight))
            add('execution_rows_metric',logits.new_tensor(1.))
            if cfg.get('rule_execution_cost',False):
                from .outcome_supervision import rule_execution_cost
                if weights.get('rule_execution',0)<=0:raise ValueError('Rule execution cost needs explicit positive weight')
                add('rule_execution',rule_execution_cost(result['probs'][None],labels))
        prior,memory,conflict=model.advance(program,result,a,row['feedback'],memory)
        target,confidence=model.transition_target(program,row['next_image'],memory,
            tokens=None if next_tokens is None else next_tokens[index:index+1])
        # Observability weights come from a stopped-gradient EMA teacher; floor prevents loss masking.
        add('transition',transition_loss(prior[0],target,result['belief'],confidence,
            balanced=cfg.get('balanced_transition_loss',False)))
        if cfg.get('discrete_reasoning',False):
            expected=target.detach()>=.5;prediction=prior[0].detach()>=.5
            copied=result['belief'].detach()>=.5;changed=expected!=copied
            bit_correct+=int((prediction==expected).sum());bit_total+=expected.numel()
            copy_correct+=int((copied==expected).sum());changed_total+=int(changed.sum())
            changed_correct+=int(((prediction==expected)&changed).sum())
            if 'symbol_codes' in row:
                saved=torch.tensor(row['symbol_codes'],device=device)
                symbol_mismatch+=int((result['symbol_codes']!=saved).sum());symbol_total+=saved.numel()
        if cfg.get('symbolic_inverse',False) and a!=0 and not row['collision'] and row['feedback'][1]>.01:
            # Infer the executed action exclusively by comparing the 17 symbolic
            # state-transition predictions. There is no RGB-to-action decoder.
            visual,visual_c=model.grounder.observe(result['tokens'],program.definitions,[])
            keep=model.grounder.retain_all(program.definitions)
            predictions,_,_=model.logic.transition_all(program,visual,visual_c,keep)
            inverse_logits=inverse_transition_logits(predictions,target,discrete=cfg.get('discrete_reasoning',False))
            add('inverse',F.cross_entropy(inverse_logits[None],action[None]))
            inverse_count+=1;inverse_correct+=int(inverse_logits.detach().argmax()==a)
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
    loss=sum(weights.get(k,1.)*v for k,v in terms.items() if not k.endswith('_metric'))
    stats={k:float(v.detach()) for k,v in terms.items()}
    if expert:stats.update(expert_action_accuracy=expert_correct/max(1,expert_count),expert_action_count=expert_count,
        expert_context_count=n-expert_count)
    if bit_total:
        stats.update(transition_bit_accuracy=bit_correct/bit_total,copy_bit_accuracy=copy_correct/bit_total,
            changed_symbol_bits=changed_total,changed_symbol_bit_accuracy=changed_correct/max(1,changed_total),
            symbol_recompute_mismatch=symbol_mismatch/max(1,symbol_total))
    if inverse_count:stats.update(inverse_accuracy=inverse_correct/inverse_count,inverse_examples=inverse_count)
    if ratios:stats['ppo_ratio_mean']=float(torch.stack(ratios).mean())
    if ratios:
        stats['ppo_ratio_max_error']=float((torch.stack(ratios)-1).abs().max())
        if cfg.get('require_exact_ppo_replay',False) and stats['ppo_ratio_max_error']>1e-4:
            raise RuntimeError('Same-update PPO replay changed sampled action likelihoods')
    if cfg.get('predicate_batch_regularization',False):stats['_observations']=torch.stack(observations)
    if return_loss_terms:stats['_loss_terms']={k:v for k,v in terms.items() if not k.endswith('_metric')}
    return loss,stats


def train_batch(model,batch,normalization=(0.,1.)):
    """Actual batched VLM programs and all frame encodings, with recurrent per-episode logic."""
    from .runtime import image_tensor
    # Keep PPO observations byte-identical to the sampled rollout. Robust
    # grounding is applied only to expert rows, including in mixed batches.
    robust=model.cfg.get('robust_grounding',False) and any(expert for _,expert in batch)
    requests=[(c['task'],c['images'],c['feedback'],c['previous']) for c,_ in batch]
    programs=model.generator.generate_many(requests)
    rows=[row for c,_ in batch for row in c['rows']]
    originals=[row['image'] for row in rows];views=originals
    if robust:
        from .augmentation import perturb_observation
        expert_rows=[expert for chunk,expert in batch for _ in chunk['rows']]
        views=[perturb_observation(image) if expert else image for image,expert in zip(originals,expert_rows)]
    images=torch.cat([image_tensor(image,model.cfg['image_size'],model.device) for image in views])
    tokens=model.grounder.encode(images)
    with torch.no_grad():
        reference_tokens=model.grounder.encode(torch.cat([image_tensor(image,model.cfg['image_size'],model.device)
            for image in originals])) if robust else None
        next_images=torch.cat([image_tensor(row['next_image'],model.cfg['image_size'],model.device) for row in rows])
        next_tokens=model.teacher.encode(next_images)
    offset=0;losses=[];statistics=[]
    for (chunk,expert),program in zip(batch,programs):
        n=len(chunk['rows'])
        loss,stats=train_chunk(model,chunk,expert,normalization,program,tokens[offset:offset+n],next_tokens[offset:offset+n],
            reference_tokens=None if reference_tokens is None or not expert else reference_tokens[offset:offset+n])
        losses.append(loss);statistics.append(stats);offset+=n
    if model.cfg.get('action_normalized_loss',False):
        sizes=tokens.new_tensor([len(c['rows']) for c,_ in batch])
        loss=(torch.stack(losses)*sizes).sum()/sizes.sum()
    else:loss=torch.stack(losses).mean()
    if model.cfg.get('predicate_batch_regularization',False):
        values=torch.cat([s.pop('_observations') for s in statistics])
        centered=values-values.mean(0)
        sd=(centered.square().mean(0)+1e-5).sqrt()
        variance=(.12-sd).clamp_min(0).mean()
        normalized=centered/sd.clamp_min(.03)
        covariance=normalized.T@normalized/max(1,len(values))
        offdiag=covariance-torch.diag(covariance.diag())
        decorrelation=offdiag.square().mean()
        loss=loss+model.cfg['loss'].get('variance',1.)*variance+model.cfg['loss'].get('decorrelation',.1)*decorrelation
        for s in statistics:s.update(predicate_std=float(sd.mean().detach()),variance=float(variance.detach()),decorrelation=float(decorrelation.detach()))
        if model.cfg.get('discrete_reasoning',False):
            from .symbols import tokenize
            # Usage regularization supplies no predicate meanings or action
            # assignments. It discourages an unused all-true/all-false alphabet.
            bits=tokenize(values).values
            frequency=bits.mean(0)
            usage=(frequency-.5).square().mean()
            loss=loss+model.cfg['loss'].get('symbol_usage',.2)*usage
            for s in statistics:s.update(symbol_usage=float(usage.detach()),
                variable_predicates=int(((frequency>0)&(frequency<1)).sum().detach()))
    return loss,statistics


def train_revision(model,trial):
    c=trial['chunk']
    predicted=model.generator(c['task'],c['images'],c['feedback'],c['previous'])
    return program_distance(predicted,trial['target'])*model.cfg['loss']['revision']
