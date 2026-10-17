"""Frozen-checkpoint diagnostics using training-only measured transitions."""
import argparse,json,random,time,os
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from predictive_nesy.data import sha
from predictive_nesy.anonymous_model import AnonymousRuleModel,Config,straight_through,reward_all
from predictive_nesy.anonymous_data import OutcomeBank
from predictive_nesy.anonymous_training import outcome_predictions


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True)
    p.add_argument('--out',required=True);p.add_argument('--batches',type=int,default=32)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    if a.device=='cuda' and os.environ.get('CUDA_VISIBLE_DEVICES') not in {'3','4','5','7'}:raise ValueError('Unauthorized GPU')
    torch.set_num_threads(2);torch.manual_seed(1540);rng=random.Random(1540)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if not checkpoint['manifest']['architecture'].startswith('anonymous_generated_rules_'):raise ValueError('Wrong architecture')
    m=AnonymousRuleModel(Config(**checkpoint['model_config'])).eval().to(a.device);m.load_state_dict(checkpoint['model'])
    bank=OutcomeBank(a.data,checkpoint['manifest']['data']['train_layouts']);records=[];begin=time.time()
    for _ in range(a.batches):
        data=bank.sample(rng,4,a.device);program,state,after=outcome_predictions(m,data);b=4;c=m.cfg
        memory=program['context_raw']
        for j in range(data['offset']):
            x=m.context_read(m.embed(data['prefix_codes'][:,j+1]))+m.action_embed(data['prefix_actions'][:,j])
            _,h=m.history(x[:,None],memory[None]);memory=h[0]
        current=m.ground_features(program,data['prefix_codes'][:,-1],memory)
        acts=torch.arange(c.actions,device=a.device)[None].expand(b,-1)
        x=m.context_read(m.embed(data['successors']))+m.action_embed(acts)
        _,h=m.history(x.reshape(b*c.actions,1,-1),memory.repeat_interleave(c.actions,0)[None])
        next_memory=h[0].reshape(b,c.actions,-1);next_memory[:,0]=memory
        defs={'definitions':program['definitions'].repeat_interleave(c.actions,0)}
        targets=m.ground_features(defs,data['successors'].flatten(0,1),next_memory.flatten(0,1)).reshape(b,c.actions,c.factors,c.states)
        predicted_score=reward_all(program,state,m.pairs);actual_score=data['scalar_rewards']
        recognized_score=reward_all(program,straight_through(current),m.pairs)
        of=c.observed_factors
        for i in range(b):
            records.append(dict(offset=data['offset'],episode_id=data['episode_ids'][i],
                current_observed_match=float((state[i,:of].argmax(-1)==current[i,:of].argmax(-1)).float().mean()),
                current_memory_match=float((state[i,of:].argmax(-1)==current[i,of:].argmax(-1)).float().mean()),
                successor_observed_match=float((after[i,:,:of].argmax(-1)==targets[i,:,:of].argmax(-1)).float().mean()),
                successor_memory_match=float((after[i,:,of:].argmax(-1)==targets[i,:,of:].argmax(-1)).float().mean()),
                state_entropy=float(-(state[i]*state[i].clamp_min(1e-8).log()).sum(-1).mean()),
                reward_rmse=float((predicted_score[i]-actual_score[i]).square().mean().sqrt()),
                recognized_reward_rmse=float((recognized_score[i]-actual_score[i]).square().mean().sqrt()),
                terminal_action_score=float(predicted_score[i,0]),measured_terminal_score=float(actual_score[i,0]),
                recognized_terminal_score=float(recognized_score[i,0]),
                movement_score=float(predicted_score[i,1:].mean())))
    groups=defaultdict(list)
    for r in records:groups[r['offset']].append(r)
    def mean(rows):
        return dict(samples=len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ['offset','episode_id']})
    report=dict(training_only=True,navigation_reruns=0,checkpoint_sha256=sha(a.checkpoint),update=checkpoint['update'],
        source_bank=bank.refresh(),overall=mean(records),by_offset={k:mean(v) for k,v in groups.items()},seconds=time.time()-begin)
    Path(a.out).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
