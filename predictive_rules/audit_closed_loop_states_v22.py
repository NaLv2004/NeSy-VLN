"""Post-execution state audit against immutable real executed symbolic proofs."""
import json,os,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
from PIL import Image
from predictive_nesy.separated_goal_world import SeparatedGoalWorld,Config
from predictive_nesy.environment import FrozenVision,make_bank
from predictive_nesy.raw_boundary_vision import RawBoundaryGraphEncoder
from predictive_nesy.gate import NeuralBoundaryGate
from predictive_nesy.graph_subgoal_interpreter import Program
from predictive_nesy.data import sha
from recover_boundary_features_v20 import Replay

@torch.no_grad()
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='7';torch.set_num_threads(2)
    root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
    origin=root/'diagnostics/seen_training_navigation_v21';out=root/'diagnostics/closed_loop_states_v22';assert not out.exists();out.mkdir()
    result=json.loads((origin/'summary.json').read_text());assert result['training_only'] and result['episodes']==32
    checkpoint=Path(result['checkpoint']);assert sha(checkpoint)==result['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);bank=make_bank()
    model=SeparatedGoalWorld(Config(**saved['model_config'])).cuda().eval();model.load_state_dict(saved['model'])
    vision=FrozenVision(saved['manifest']['frozen_vision']);gate=NeuralBoundaryGate([model,vision.vision]);encoder=RawBoundaryGraphEncoder(vision,gate)
    sim=Replay(bank,out);episodes=[];began=time.time()
    try:
        for file in sorted(origin.glob('*/trace.json')):
            trace=json.loads(file.read_text());folder=file.parent;episode=trace['episode'];evaluation=json.loads((folder/'evaluation.json').read_text())
            assert episode['scene'] in bank.provenance['train_layouts']
            actions=[s['action'] for s in trace['steps']];frames=sim.call('replay',episode=episode,actions=actions)['rgb'];assert len(frames)==len(actions)+1
            image_errors={}
            for name,index in [('first',0),('last',-1)]:
                expected=np.asarray(Image.open(folder/(name+'.png')).convert('RGB').resize((224,224),Image.Resampling.BILINEAR))
                image_errors[name]=float(np.abs(expected.astype(np.float32)-frames[index].astype(np.float32)).mean())
            assert max(image_errors.values())==0.,(episode['episode_id'],image_errors)
            with gate.boundary(0):codes,raw=encoder.encode_joint([Image.fromarray(frame) for frame in frames])
            np.savez_compressed(out/(episode['episode_id']+'.npz'),codes=codes.cpu().numpy().astype(np.uint8),actions=np.asarray(actions))
            events=[];previous=[17]+actions
            for event in trace['programs']:
                start=event['start'];n=min(len(event['plan']['actions']),len(actions)-start)
                program=Program.load(folder/event['file']);width=min(start+1,11);begin=max(0,start-10)
                with gate.boundary(start):
                    p=model.generate(torch.tensor([['chair','table'].index(episode['category'])],device='cuda'),codes[None,begin:start+1],
                        torch.tensor([previous[begin:start+1]],device='cuda'),torch.tensor([width],device='cuda'),start=start,boundary_features=raw[None,begin:start+1])
                    p['definitions']=torch.tensor(program.definitions.copy(),device='cuda')[None]
                    posterior=model.encode_sequence(p,codes[None,start:start+n+1])[0][0]
                labels=posterior.argmax(-1).cpu().tolist();table=program.transitions.argmax(-1)
                expected=[event['initial_state'][0]]+[s['after'][0] for s in trace['steps'][start:start+n]]
                initial_match=labels[0]==expected[0];steps=[]
                for t in range(n):
                    proof=trace['steps'][start+t];action=proof['action'];assert table[expected[t],action]==expected[t+1]
                    probabilities=program.transitions[expected[t],action];pred=expected[t+1]
                    rank=int((posterior[t+1]>posterior[t+1,pred]).sum())+1
                    steps.append(dict(step=start+t,action=action,predicted_before=expected[t],observed_before=labels[t],predicted_after=pred,
                        observed_after=labels[t+1],autonomous_correct=pred==labels[t+1],one_step_correct=int(table[labels[t],action])==labels[t+1],
                        selected_rule_probability=float(probabilities[pred]),observed_probability_under_selected_rule=float(probabilities[labels[t+1]]),
                        predicted_label_posterior_rank=rank,unchanged_rgb_codes=bool(torch.equal(codes[start+t],codes[start+t+1])),
                        collision_for_report_only=proof.get('collision_for_report_only',False)))
                events.append(dict(start=start,initial_grounding_agrees=initial_match,plan_found=event['plan']['found'],subgoal=int(program.subgoal[0]),
                    actions=actions[start:start+n],predicted_states=expected,observed_states=labels,
                    first_divergence=next((i+1 for i,x in enumerate(steps) if not x['autonomous_correct']),None),
                    observed_endpoint_matches_goal=labels[-1]==int(program.subgoal[0]),fully_executed_plan=n>0 and n==len(event['plan']['actions']),steps=steps))
            episodes.append(dict(episode=episode,evaluation=evaluation,image_errors=image_errors,events=events))
            if len(episodes)%8==0:print(json.dumps(dict(completed=len(episodes),total=32,seconds=time.time()-began)),flush=True)
        groups={};by_first=Counter();stops=Counter()
        for label in ['all','successful','failed']:
            er=[r for r in episodes if label=='all' or bool(r['evaluation']['success'])==(label=='successful')]
            events=[v for r in er for v in r['events']];usable=[v for v in events if v['initial_grounding_agrees']]
            steps=[s for v in usable for s in v['steps']];executed=[v for v in usable if v['steps']]
            groups[label]=dict(episodes=len(er),programs=len(events),initial_grounding_agreements=len(usable),steps=len(steps),
                autonomous_correct=sum(s['autonomous_correct'] for s in steps),one_step_correct=sum(s['one_step_correct'] for s in steps),
                first_transition_correct=sum(v['steps'][0]['autonomous_correct'] for v in executed),executed_programs=len(executed),
                all_executed_states_correct=sum(all(s['autonomous_correct'] for s in v['steps']) for v in executed),
                observed_endpoint_reaches_subgoal=sum(v['observed_endpoint_matches_goal'] for v in executed),
                predicted_state_top5=sum(s['predicted_label_posterior_rank']<=5 for s in steps),
                median_predicted_state_rank=float(np.median([s['predicted_label_posterior_rank'] for s in steps])) if steps else None,
                mean_rule_probability_correct=float(np.mean([s['selected_rule_probability'] for s in steps if s['autonomous_correct']])) if any(s['autonomous_correct'] for s in steps) else None,
                mean_rule_probability_wrong=float(np.mean([s['selected_rule_probability'] for s in steps if not s['autonomous_correct']])) if any(not s['autonomous_correct'] for s in steps) else None)
            if label=='all':
                by_first.update(str(v['first_divergence']) for v in executed)
                for r in er:
                    if not r['events']:continue
                    v=r['events'][-1]
                    if not v['steps'] or v['actions'][-1]!=0:continue
                    s=v['steps'][-1];stops['STOPs']+=1;stops['successful']+=bool(r['evaluation']['success'])
                    stops['observed_state_matches_subgoal']+=v['observed_endpoint_matches_goal']
                    stops['STOP_rule_correct_given_actual_state']+=s['one_step_correct']
                    stops['wrong_task_completion_despite_observed_subgoal']+=(not r['evaluation']['success'] and v['observed_endpoint_matches_goal'])
        report=dict(training_only=True,post_execution_only=True,no_new_policy_rollouts=True,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            exact_first_last_replay_images=True,groups=groups,first_divergence_histogram=dict(by_first),STOP_diagnosis=dict(stops),episodes=episodes,seconds=time.time()-began)
        (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='episodes'}),flush=True)
    finally:gate.close();sim.close()

if __name__=='__main__':main()
