"""Training-only paired counterfactual program trials. Never called by evaluation."""
import torch
from .runtime import pil,public_feedback
from .logic import NoSupportedAction


@torch.no_grad()
def compare_candidates(model,sim,bank,episode,policy,rng):
    assert episode['episode_id'] in {x['episode_id'] for x in bank.train}
    chunks=[c for c in policy.chunks if c['previous'] is not None and c['rows']]
    if not chunks:return None
    chunk=rng.choice(chunks)
    prefix=[row['action'] for c in policy.chunks for row in c['rows']][:chunk['start']]
    hidden=model.generator.context(chunk['task'],chunk['images'],chunk['feedback'],chunk['previous'])
    candidates=[chunk['previous'],chunk['program']]
    for _ in range(model.cfg['candidate_count']-1):
        p=model.cfg['program']
        noise=torch.randn(p['k']+p['r'],p['d'],device=model.device)*.08
        candidates.append(model.generator.head(hidden,chunk['previous'],noise).detach())
    scores=[]; outcomes=[]
    for candidate in candidates:
        response=sim.call('reset',episode=episode)
        for action in prefix:
            response=sim.call('step',action=action)
            if response['done']:raise RuntimeError('Non-deterministic training prefix replay')
        image=pil(response['rgb']); memory=list(chunk['memory']); prior=None; total=0.; actions=[]
        for _ in range(min(model.cfg['candidate_horizon'],model.cfg['horizon']-len(prefix))):
            try:result=model.step(candidate,image,memory,prior)
            except NoSupportedAction:total-=40;break
            action=int(result['probs'].argmax()); actions.append(action)
            response=sim.call('step',action=action); after=pil(response['rgb'])
            _,feedback=public_feedback(response,image,after)
            prior,memory,_=model.advance(candidate,result,action,feedback,memory)
            total+=float(response['reward']);image=after
            if response['done']:break
        scores.append(total)
        outcomes.append({'success':bool(response.get('metrics',{}).get('success',False)),
                         'actions':actions})
    winner=max(range(len(scores)),key=scores.__getitem__)
    return {'chunk':chunk,'target':candidates[winner].detach(),'winner':winner,'scores':scores,
            'outcomes':outcomes,'gain_over_previous':scores[winner]-scores[0],
            'gain_over_default_revision':scores[winner]-scores[1],
            'episode_id':episode['episode_id'],'start':chunk['start']}
