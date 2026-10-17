"""VLM-conditioned continuous program synthesis and revision; no action bypass."""
import json
import torch
from torch import nn
from .program import Program,describe


class ProgramDecoder(nn.Module):
    def __init__(self,hidden=2048,k=16,d=128,r=48,a=17):
        super().__init__(); self.k,self.d,self.r,self.a=k,d,r,a
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

    def forward(self,hidden,previous=None,noise=None):
        h=self.context(hidden.float())
        if h.shape[1]>256:
            h=torch.nn.functional.adaptive_avg_pool1d(h.transpose(1,2),256).transpose(1,2)
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
        rule=self.decoder(rq[None],torch.cat([h,definitions[None]],1)).squeeze(0)
        literals=torch.cat([self.constant[None],definitions+self.sign[0],definitions+self.sign[1]],0)
        logits=torch.einsum('rjd,kd->rjk',self.premise(rule).reshape(self.r,2,self.d),literals)/self.d**.5
        # Every action rule must consult at least one nonconstant literal.
        n=self.r//6; logits=logits.clone(); logits[2*n:4*n,0,0]=-1e9
        premises=logits.softmax(-1)
        state=(self.state(rule)@definitions.T/self.d**.5).softmax(-1)
        effects=torch.einsum('rjd,kd->rjk',self.effect(rule).reshape(self.r,2,self.d),definitions)
        effects=(effects.flatten(1)/self.d**.5).softmax(-1)
        return Program(definitions,premises,state,self.act(rule).softmax(-1),effects,
                       self.condition(rule).softmax(-1),self.weight(rule).squeeze(-1).sigmoid(),
                       0 if previous is None else previous.version+1)


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
    for key in ['premises','state_heads','action_heads','effects','action_conditions']:
        p,q=getattr(student,key),getattr(teacher,key).detach()
        loss=loss+torch.nn.functional.kl_div(p.clamp_min(1e-8).log(),q,reduction='batchmean')
    return loss+(student.weights-teacher.weights.detach()).square().mean()
