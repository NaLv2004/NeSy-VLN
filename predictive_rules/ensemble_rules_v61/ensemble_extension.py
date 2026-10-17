"""Compile bootstrap observation rules in one unchanged symbolic vocabulary."""
import copy
SingleWorld=FactoredWorld
class FactoredWorld(SingleWorld):
    def __init__(self,cfg=None):
        super().__init__(cfg)
        self.member_rules=nn.ModuleList([copy.deepcopy(self.definition_rule) for _ in range(3)])
    def load_candidate(self,source,heads):
        missing,unexpected=self.load_state_dict(source,strict=False)
        assert not unexpected and set(missing)=={k for k in self.state_dict() if k.startswith('member_rules.')}
        assert len(heads)==4  # head0 is the separately measured single control
        for module,weights in zip(self.member_rules,heads[1:]):module.load_state_dict(weights,strict=True)
    @torch.no_grad()
    def compile(self,context,task):
        assert context.shape[0]==1
        c=self.cfg
        m,o,a=torch.meshgrid(torch.arange(c.memories,device=context.device),
            torch.arange(c.observations,device=context.device),torch.arange(c.actions,device=context.device),indexing='ij')
        m,o,a=m.flatten(),o.flatten(),a.flatten();mdefs=self.definitions(context)
        memories=[];outcomes=[[] for _ in self.member_rules]
        for begin in range(0,len(m),4096):
            mm=F.one_hot(m[None,begin:begin+4096],c.memories).float()
            oo=F.one_hot(o[None,begin:begin+4096],c.observations).float();aa=a[None,begin:begin+4096]
            next_m,_=BaseWorld.transition(self,context,mm,oo,aa,mdefs)
            memories.append(next_m.argmax(-1)[0].cpu())
            inputs=definition_inputs(self,context,mm,oo,aa)
            preserve=inputs[-1].bool()
            for output,head in zip(outcomes,self.member_rules):
                p=head(*inputs).exp();p=torch.where(preserve,oo,p);p=p/p.sum(-1,keepdim=True)
                output.append(p[0].cpu())
        gm,go=self.goal_logits(context,task)
        return dict(next_memory=torch.cat(memories).reshape(c.memories*c.observations,c.actions).numpy(),
            outcome_probability=torch.stack([torch.cat(x).reshape(c.memories*c.observations,c.actions,c.observations) for x in outcomes]).numpy(),
            goal=int(gm.argmax(-1))*c.observations+int(go.argmax(-1)))
