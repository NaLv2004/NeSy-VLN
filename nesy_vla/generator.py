"""VLM-conditioned continuous program synthesis and revision; no action bypass."""
import json,math
import torch
from torch import nn
from .program import Program,describe


def pool_context(hidden,bins=256):
    """Exact adaptive interval means without CUDA's long-axis pool backward kernel."""
    length=hidden.shape[1]
    if length<=bins:return hidden
    index=torch.arange(bins,device=hidden.device)
    starts=torch.div(index*length,bins,rounding_mode='floor')
    ends=torch.div((index+1)*length+bins-1,bins,rounding_mode='floor')
    prefix=torch.nn.functional.pad(hidden.cumsum(1),(0,0,1,0))
    return (prefix[:,ends]-prefix[:,starts])/(ends-starts)[None,:,None]


def sparse_literals(logits,k,top=2):
    """Learned two-literal mixtures; a predicate and its negation cannot cancel.

    The forward program remains a convex soft mixture. A straight-through dense
    surrogate lets currently unselected connections learn; no semantic labels or
    action-specific predicate assignments are supplied.
    """
    dense=(logits/.7).softmax(-1)
    first=logits.argmax(-1,keepdim=True)
    mask=torch.zeros_like(logits).scatter_(-1,first,1.)
    remaining=logits.masked_fill(mask.bool(),-1e9)
    opposite=torch.where(first==0,first,torch.where(first<=k,first+k,first-k))
    remaining=remaining.scatter(-1,opposite,-1e9)
    second=remaining.argmax(-1,keepdim=True)
    mask=mask.scatter(-1,second,1.)
    sparse=(logits/.7).masked_fill(~mask.bool(),-1e9).softmax(-1)
    return sparse.detach()+dense-dense.detach()


def balanced_routes(logits,iterations=20):
    """Soft learned rule-to-action matching, with equal column capacity."""
    z=logits.float()/.35
    for _ in range(iterations):
        z=z-torch.logsumexp(z,-1,keepdim=True)
        z=z-torch.logsumexp(z,0,keepdim=True)
    return z.softmax(-1)


class ProgramDecoder(nn.Module):
    def __init__(self,hidden=2048,k=16,d=128,r=48,a=17,structured=False,multi_effects=False,action_semantics='horn',route_temperature=1.,conjunction_operator='product',neural_conjunction_gates=False):
        super().__init__(); self.k,self.d,self.r,self.a=k,d,r,a;self.structured=structured
        self.multi_effects=multi_effects
        self.action_semantics=action_semantics
        if conjunction_operator not in ('product','minimum'):raise ValueError('Unknown conjunction operator')
        self.conjunction_operator=conjunction_operator
        if type(neural_conjunction_gates) is not bool:raise ValueError('Neural conjunction gate flag must be Boolean')
        if neural_conjunction_gates and action_semantics!='conjunctive_rules':raise ValueError('Gates require conjunction rules')
        self.neural_conjunction_gates=neural_conjunction_gates
        if not math.isfinite(route_temperature) or route_temperature<=0:raise ValueError('Positive finite route temperature required')
        self.route_temperature=float(route_temperature)
        if action_semantics=='branch_tree':
            leaves=r//3
            if r%6 or leaves&(leaves-1):raise ValueError('Branch tree requires a power-of-two action-rule count')
        self.context=nn.Linear(hidden,d)
        self.pred_queries=nn.Parameter(torch.randn(k,d)*.03)
        self.rule_queries=nn.Parameter(torch.randn(r,d)*.03)
        self.decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(d,4,4*d,dropout=0.,
            batch_first=True,norm_first=True),2)
        self.definition=nn.Sequential(nn.Linear(d,d),nn.Tanh())
        self.previous_rule=nn.Linear(2*(2*k+1)+k+a+2*k+a+1,d)
        self.premise=nn.Linear(d,2*d); self.sign=nn.Parameter(torch.randn(2,d)*.01)
        self.constant=nn.Parameter(torch.randn(d)*.01)
        self.state=nn.Linear(d,d); self.act=nn.Linear(d,a)
        self.effect=nn.Linear(d,2*d); self.condition=nn.Linear(d,a)
        self.weight=nn.Linear(d,1)
        if structured:
            nn.init.orthogonal_(self.pred_queries,gain=d**.5)
            nn.init.orthogonal_(self.rule_queries,gain=d**.5)
            self.connection_logits=nn.Parameter(torch.randn(r,2,2*k+1)*.5)
            self.routing_logits=nn.Parameter(torch.randn(r,a)*.5)

    def forward(self,hidden,previous=None,noise=None):
        h=self.context(hidden.float())
        h=pool_context(h)
        pq=self.pred_queries
        rq=self.rule_queries
        if previous is not None:
            pq=pq+previous.definitions.to(pq)
            old=torch.cat([previous.premises.flatten(1),previous.state_heads,previous.action_heads,
                           previous.effects,previous.action_conditions,previous.weights[:,None]],-1).to(rq)
            rq=rq+self.previous_rule(old)
        if noise is not None:
            pq=pq+noise[:self.k]; rq=rq+noise[self.k:]
        predicates=self.decoder(pq[None],h).squeeze(0)
        definitions=self.definition(predicates)
        if self.structured:
            definitions=torch.nn.functional.layer_norm(self.pred_queries+.35*definitions,(self.d,))
        rule=self.decoder(rq[None],torch.cat([h,definitions[None]],1)).squeeze(0)
        literals=torch.cat([self.constant[None],definitions+self.sign[0],definitions+self.sign[1]],0)
        logits=torch.einsum('rjd,kd->rjk',self.premise(rule).reshape(self.r,2,self.d),literals)/self.d**.5
        if self.structured:logits=logits+self.connection_logits
        # Every action rule must consult at least one nonconstant literal.
        n=self.r//6; logits=logits.clone(); logits[2*n:4*n,0,0]=-1e9
        if self.action_semantics=='branch_tree':logits[:2*n-1,0,0]=-1e9
        premises=sparse_literals(logits,self.k) if self.structured else logits.softmax(-1)
        state=(self.state(rule)@definitions.T/self.d**.5).softmax(-1)
        effects=torch.einsum('rjd,kd->rjk',self.effect(rule).reshape(self.r,2,self.d),definitions)
        effects=effects/self.d**.5
        conjunctions=None
        if self.action_semantics=='conjunctive_rules':
            # The shared rule/definition compatibility head predicts soft
            # true/false/unused requirements. Action rows were not used as
            # effect rules; their conditions now participate explicitly.
            conjunctions=torch.cat([torch.zeros_like(effects[:,:1]),effects],1).softmax(1).transpose(1,2)
        if self.multi_effects:
            # Each learned predicate chooses unchanged/add/delete independently.
            # One action may change several predicates, as in an ordinary STRIPS effect set.
            choices=torch.cat([torch.zeros_like(effects[:,:1]),effects],1).softmax(1)
            effects=choices[:,1:].flatten(1)
        else:effects=effects.flatten(1).softmax(-1)
        action_heads=self.act(rule).softmax(-1)
        weights=self.weight(rule).squeeze(-1).sigmoid()
        if self.structured:
            routes=balanced_routes((self.act(rule)+self.routing_logits)[2*n:4*n])
            if self.route_temperature!=1:
                # Calibrate generated rule conclusions, never an image-action
                # head or an outside-the-proof fallback. Recorded programs
                # retain the exact calibrated conclusion probabilities.
                routes=(routes.clamp_min(1e-30).log()/self.route_temperature).softmax(-1)
            action_heads=torch.cat([action_heads[:2*n],routes,action_heads[4*n:]],0)
            weights=torch.cat([.2*weights[:2*n],weights[2*n:]],0)
        return Program(definitions,premises,state,action_heads,effects,
                       self.condition(rule).softmax(-1),weights,
                       0 if previous is None else previous.version+1,
                       'per_predicate' if self.multi_effects else 'categorical',self.action_semantics,conjunctions,self.conjunction_operator,self.neural_conjunction_gates)


class VLMProgramGenerator(nn.Module):
    def __init__(self,model_path,cfg):
        super().__init__()
        from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
        from peft import LoraConfig,get_peft_model
        self.processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
        base=Qwen3_5ForConditionalGeneration.from_pretrained(model_path,local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa')
        self.vlm=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,
            target_modules=['q_proj','v_proj','o_proj'],task_type='CAUSAL_LM'))
        self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
        self.vlm.enable_input_require_grads()
        self.head=ProgramDecoder(hidden=base.config.text_config.hidden_size,**cfg)

    def prompt(self,task,feedback,previous):
        prefix=('Construct an executable soft predicate program for a robot navigation task. '
            'Predicate meanings and rule connections must be learned, not a supplied vocabulary. '
            'The attached images are chronological observed history; the final image is current. '
            'Use actual executed actions and observed outcomes. Revise the previous program after ten executed actions. '
            'Do not output a direct action. Task: '+task+'\n')
        if previous is not None:
            p=previous.cpu()
            compact=[[r,*p.premises[r].argmax(-1).tolist(),int(p.state_heads[r].argmax()),
                int(p.action_heads[r].argmax()),int(p.effects[r].argmax()),round(float(p.weights[r]),2)]
                for r in range(p.weights.numel())]
            prefix+='Previous rules [id, premise1, premise2, state head, action head, effect head, weight]: '+json.dumps(compact,separators=(',',':'))+'\n'
        return prefix+'Executed evidence: '+json.dumps(feedback[-10:],separators=(',',':'))

    def context_many(self,requests):
        conversations=[]
        for task,images,feedback,previous in requests:
            selected=[images[0]] if len(images)==1 else [images[0],images[-1]] if len(images)<3 else [images[0],images[-2],images[-1]]
            content=[{'type':'image','image':im.resize((320,240))} for im in selected]
            content.append({'type':'text','text':self.prompt(task,feedback,previous)})
            conversations.append([{'role':'user','content':content}])
        self.processor.tokenizer.padding_side='left'
        inputs=self.processor.apply_chat_template(conversations,tokenize=True,padding=True,
            add_generation_prompt=True,return_dict=True,return_tensors='pt')
        inputs={k:v.to(next(self.vlm.parameters()).device) if torch.is_tensor(v) else v for k,v in inputs.items()}
        # Call the multimodal body: no vocabulary-sized LM logits and no token action decoder.
        outputs=self.vlm.get_base_model().model(**inputs,use_cache=False,return_dict=True)
        hidden=outputs.last_hidden_state
        return [hidden[i:i+1,inputs['attention_mask'][i].bool()] for i in range(len(requests))]

    def context(self,task,images,feedback,previous=None):
        return self.context_many([(task,images,feedback,previous)])[0]

    def generate_many(self,requests):
        contexts=self.context_many(requests)
        return [self.head(h,req[3]) for h,req in zip(contexts,requests)]

    def forward(self,task,images,feedback,previous=None,noise=None):
        hidden=self.context(task,images,feedback,previous)
        return self.head(hidden,previous,noise)


def program_distance(student,teacher):
    loss=(student.definitions-teacher.definitions.detach()).square().mean()
    if student.effect_mode!=teacher.effect_mode:raise ValueError('Cannot distill incompatible effect semantics')
    if student.action_semantics!=teacher.action_semantics:raise ValueError('Cannot distill incompatible action semantics')
    for key in ['premises','state_heads','action_heads','action_conditions']+(['effects'] if student.effect_mode=='categorical' else []):
        p,q=getattr(student,key),getattr(teacher,key).detach()
        loss=loss+torch.nn.functional.kl_div(p.clamp_min(1e-8).log(),q,reduction='batchmean')
    if student.effect_mode=='per_predicate':
        def triplets(p):
            add,delete=p.effects.chunk(2,-1)
            return torch.stack([1-add-delete,add,delete],-1).clamp_min(1e-8)
        loss=loss+torch.nn.functional.kl_div(triplets(student).log(),triplets(teacher).detach(),reduction='none').sum(-1).mean()
    if student.conjunctive_conditions is not None:
        loss=loss+torch.nn.functional.kl_div(student.conjunctive_conditions.clamp_min(1e-8).log(),
            teacher.conjunctive_conditions.detach(),reduction='none').sum(-1).mean()
    return loss+(student.weights-teacher.weights.detach()).square().mean()
