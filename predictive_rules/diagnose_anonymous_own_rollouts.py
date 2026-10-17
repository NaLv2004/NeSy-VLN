"""Posthoc training-only diagnosis: predicted versus observed anonymous states.

No environment execution occurs. Recorded future images are labels only. The
checkpoint must exactly match the collection policy to attribute its errors.
"""
import argparse,json,os,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from predictive_nesy.data import read,sha
from predictive_nesy.anonymous_data import OutcomeBank
from predictive_nesy.anonymous_model import AnonymousRuleModel,Config,straight_through,reward_all
from predictive_nesy.anonymous_training import outcome_predictions


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--data',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=parser.parse_args()
    if args.device=='cuda' and os.environ.get('CUDA_VISIBLE_DEVICES') not in {'3','4','5','7'}:raise ValueError('Unauthorized GPU')
    torch.set_num_threads(2);saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    root=Path(args.data);manifest=read(root/'manifest.json')
    if manifest['policy_checkpoint']['sha256']!=sha(args.checkpoint):raise ValueError('Checkpoint differs from collection policy')
    if not manifest.get('training_only'):raise ValueError('Training-only data required')
    bank=OutcomeBank(root,manifest['data']['train_layouts']);bank_snapshot=bank.refresh()
    bank_snapshot['episode_ids']=sorted(bank.episodes)
    model=AnonymousRuleModel(Config(**saved['model_config'])).to(args.device).eval()
    model.load_state_dict(saved['model']);records=[];began=time.time()
    for (start,offset),selections in sorted(bank.by_key.items()):
        for begin in range(0,len(selections),4):
            rows=[];names=selections[begin:begin+4]
            for name,t in names:
                d=bank.episodes[name];history_begin=max(0,start-10)
                rows.append(dict(task=int(d['task']),history=d['codes'][history_begin:start+1],
                    previous_actions=np.array([int(d['actions'][j-1]) if j else 17 for j in range(history_begin,start+1)]),
                    prefix_codes=d['codes'][start:t+1],prefix_actions=d['actions'][start:t],successors=d['successors'][t],
                    scalar_rewards=d['scalar_rewards'][t]))
            data={key:torch.as_tensor(np.asarray([r[key] for r in rows]),device=args.device) for key in rows[0]}
            for key in ['task','history','previous_actions','prefix_codes','prefix_actions','successors']:data[key]=data[key].long()
            data.update(start=start,offset=offset,episode_ids=[name for name,t in names])
            program,predicted,_=outcome_predictions(model,data);memory=program['context_raw']
            for j in range(offset):
                x=model.context_read(model.embed(data['prefix_codes'][:,j+1]))+model.action_embed(data['prefix_actions'][:,j])
                _,h=model.history(x[:,None],memory[None]);memory=h[0]
            recognized=straight_through(model.ground_features(program,data['prefix_codes'][:,-1],memory))
            predicted_scores=reward_all(program,predicted,model.pairs);recognized_scores=reward_all(program,recognized,model.pairs)
            for i,(name,t) in enumerate(names):
                action=int(bank.episodes[name]['actions'][t]);actual=float(data['scalar_rewards'][i,action])
                records.append(dict(episode_id=name,step=t,action=action,offset=offset,actual_score=actual,
                    predicted_score=float(predicted_scores[i,action]),recognized_score=float(recognized_scores[i,action]),
                    predicted_stop_score=float(predicted_scores[i,0]),recognized_stop_score=float(recognized_scores[i,0]),
                    measured_stop_score=float(data['scalar_rewards'][i,0]),
                    visual_category_match=float((predicted[i,:4].argmax(-1)==recognized[i,:4].argmax(-1)).float().mean()),
                    memory_category_match=float((predicted[i,4:].argmax(-1)==recognized[i,4:].argmax(-1)).float().mean())))
    def aggregate(rows):
        if not rows:return dict(samples=0)
        return dict(samples=len(rows),mean_predicted_score=float(np.mean([r['predicted_score'] for r in rows])),
            mean_recognized_score=float(np.mean([r['recognized_score'] for r in rows])),
            mean_actual_score=float(np.mean([r['actual_score'] for r in rows])),
            predicted_rmse=float(np.mean([(r['predicted_score']-r['actual_score'])**2 for r in rows])**.5),
            recognized_rmse=float(np.mean([(r['recognized_score']-r['actual_score'])**2 for r in rows])**.5),
            visual_category_match=float(np.mean([r['visual_category_match'] for r in rows])),
            memory_category_match=float(np.mean([r['memory_category_match'] for r in rows])))
    failed=[r for r in records if r['action']==0 and r['actual_score']<0]
    report=dict(checkpoint_sha256=sha(args.checkpoint),training_only=True,navigation_reruns=0,bank=bank_snapshot,seconds=time.time()-began,
        all=aggregate(records),stops=aggregate([r for r in records if r['action']==0]),failed_stops=aggregate(failed),
        failed_stops_prediction_positive=sum(r['predicted_score']>0 for r in failed),
        failed_stops_observation_reverses_sign=sum(r['predicted_score']>0 and r['recognized_score']<=0 for r in failed),
        failed_stops_both_positive=sum(r['predicted_score']>0 and r['recognized_score']>0 for r in failed),
        by_offset={i:aggregate([r for r in records if r['offset']==i]) for i in range(10)},
        label_note='Observed future states are used only for offline error attribution; never fed to the collection controller')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(report,indent=2));(out/'records.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
