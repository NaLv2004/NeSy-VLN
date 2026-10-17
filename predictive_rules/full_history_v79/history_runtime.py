import torch
from torch.nn import functional as F
from history_model import WINDOW

@torch.no_grad()
def root_rows(model,predictor,program,context,features,actions,memory):
    # Reinterpret each historical observation in the CURRENT fixed rule basis.
    # This updates observation IDs only; the current memory is never regrounded.
    history=torch.stack(features[-WINDOW:])[None]
    observations=model.observe(history,context).argmax(-1)
    previous=torch.tensor([[17]+actions],device=context.device)[:,-observations.shape[1]:]
    mask=torch.ones_like(observations,dtype=torch.bool)
    defs=F.normalize(model.observation_definitions[None]+model.observation_definition_delta(context),dim=-1)
    ids=torch.arange(17,device=context.device);current=int(observations[0,-1]);state=int(memory)*64+current
    lp=torch.tensor(program.outcome_probability[state],device=context.device).clamp_min(1e-30).log()
    logp=predictor(observations.expand(17,-1),previous.expand(17,-1),mask.expand(17,-1),defs.expand(17,-1,-1),
        model.definitions(context)[:,memory].expand(17,-1),model.action_embedding(ids),context.expand(17,-1),lp,
        model.operator_class(ids).argmax(-1)==1)
    p=logp.exp();preserve=model.operator_class(ids).argmax(-1)==1
    p=torch.where(preserve[:,None],F.one_hot(torch.full_like(ids,current),64).float(),p)
    return program.next_memory[state],p.cpu().numpy(),dict(observations=observations[0].tolist(),previous_actions=previous[0].tolist())
