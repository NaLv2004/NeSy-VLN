"""Fixed TRAIN-only diagnosis; no simulator, optimizer, actor, or runtime changes.

Actual future codes below are oracle diagnostic labels, never planner inputs in
deployment. Compare generated vs recorded goals to separate goal and rule error.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from model import FactoredWorld, Config, hard
from planner import Program, plan
from data import ROOT, sha


def main():
    torch.set_num_threads(4)
    run=ROOT/'run_induced_effects_v39'
    commit=json.loads((run/'latest.COMMITTED.json').read_text())
    checkpoint=run/commit['filename']
    assert sha(checkpoint)==commit['sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).eval()
    model.load_state_dict(saved['model']);manifest=saved['manifest'];del saved
    source=Path(manifest['data']['source'])
    assert sha(source/'training_rows.json')==manifest['data']['rows_sha256']
    rows=json.loads((source/'training_rows.json').read_text())
    features=np.load(ROOT/'prequant_visual_features_v18/features.npy',mmap_mode='r')
    episodes=defaultdict(list)
    for i,row in enumerate(rows):episodes[row['episode_id']].append(i)
    selected=sorted((name for name,ids in episodes.items() if len(ids)>=5),
                    key=lambda name:hashlib.sha256(name.encode()).hexdigest())[:64]
    cases=[];began=time.time()
    with torch.inference_mode():
        for name in selected:
            ids=episodes[name];actions=torch.tensor([[int(rows[i]['label']) for i in ids]])
            assert actions[0,-1]==0
            n=len(ids)
            x=torch.from_numpy(np.array(features[ids+[ids[-1]]],copy=True))[None]
            previous=torch.cat([torch.tensor([[17]]),actions],1)
            task=torch.tensor([['chair','table'].index(rows[ids[0]]['category'])])
            ol=model.observe(x);o=hard(ol);oi=ol.argmax(-1)
            ml,_,_=model.memory_sequence(o,previous);m=hard(ml);mi=ml.argmax(-1)
            contexts=model.context_sequence(x,o,previous,task)
            actual_goal=int(mi[0,n])*model.cfg.observations+int(oi[0,n])
            for start in (0,n//2):
                context=contexts[:,start]
                exported=model.compile(context,task)
                program=Program(**exported,observations=model.cfg.observations)
                proposed=plan(program,int(mi[0,start]),int(oi[0,start]),horizon=32-start)
                oracle=Program(program.next_state,program.confidence,actual_goal,program.observations)
                oracle_proof=plan(oracle,int(mi[0,start]),int(oi[0,start]),horizon=32-start)
                current=int(mi[0,start])*program.observations+int(oi[0,start])
                executed_states=[current]
                for action in actions[0,start:].tolist():
                    current=int(program.next_state[current,action]);executed_states.append(current)
                true_states=(mi[0,start:n+1]*program.observations+oi[0,start:n+1]).numpy()
                pm,po=model.transition(context,m[:,start:n],o[:,start:n],actions[:,start:n])
                actual_next=oi[:,start+1:n+1]
                ranks=(po>po.gather(-1,actual_next[...,None])).sum(-1)+1
                true_probability=po.exp().gather(-1,actual_next[...,None])[...,0]
                stop_image=set(program.next_state[:,0].tolist())
                cases.append(dict(episode=name,start=start,remaining=n-start,task=int(task),
                    generated_goal=program.goal,actual_goal=actual_goal,
                    goal_memory_correct=program.goal//program.observations==actual_goal//program.observations,
                    goal_observation_correct=program.goal%program.observations==actual_goal%program.observations,
                    generated_goal_has_stop_preimage=program.goal in stop_image,
                    actual_goal_has_stop_preimage=actual_goal in stop_image,
                    generated_goal_plan=proposed['found'],actual_goal_plan=oracle_proof['found'],
                    teacher_sequence_reaches_generated_goal=executed_states[-1]==program.goal,
                    teacher_sequence_reaches_actual_goal=executed_states[-1]==actual_goal,
                    teacher_sequence_all_states_correct=bool(np.array_equal(executed_states,true_states)),
                    teacher_path_joint_accuracy=float((np.asarray(executed_states[1:])==true_states[1:]).mean()),
                    teacher_forced_observation_ranks=ranks[0].tolist(),
                    teacher_forced_true_observation_probability=true_probability[0].tolist(),
                    first_action_matches_teacher=bool(proposed['found'] and proposed['actions'][0]==int(actions[0,start])),
                    proposed_actions=proposed['actions'],teacher_actions=actions[0,start:].tolist()))
    summary={}
    for label,subset in [('all',cases),('initial',[c for c in cases if c['start']==0]),('midpoint',[c for c in cases if c['start']>0])]:
        entry=dict(cases=len(subset))
        for key,value in subset[0].items():
            if isinstance(value,bool):entry[key]=sum(c[key] for c in subset)/len(subset)
        ranks=[r for case in subset for r in case['teacher_forced_observation_ranks']]
        entry['teacher_forced_observation_topk']={str(k):sum(r<=k for r in ranks)/len(ranks) for k in (1,2,4,8)}
        entry['teacher_path_joint_accuracy']=float(np.mean([c['teacher_path_joint_accuracy'] for c in subset]))
        entry['generated_missing_only_goal_wrong']=sum(not c['generated_goal_plan'] and c['actual_goal_plan'] for c in subset)
        summary[label]=entry
    result=dict(time=time.time(),seconds=time.time()-began,checkpoint=commit,training_only=True,
                source_rows_sha256=manifest['data']['rows_sha256'],
                diagnostic_only='Recorded endpoints are used only to localize failure, never deployed as goal inputs.',
                summary=summary,cases=cases)
    output=run/f'reachability_audit_{commit["update"]:06d}.json'
    output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(output=str(output),summary=summary,seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
