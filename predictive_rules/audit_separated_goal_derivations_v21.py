"""Train-only proof diagnosis. Future labels never enter the program generator."""
import json,random,os
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import torch
from predictive_nesy.separated_goal_world import SeparatedGoalWorld as RawBoundaryWorld,Config,export_program
from predictive_nesy.raw_boundary_data import RawBoundaryFeatures
from predictive_nesy.subgoal_data import load_windows,batch
from predictive_nesy.graph_subgoal_interpreter import plan
from predictive_nesy.data import sha

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')

@torch.no_grad()
def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='7';torch.set_num_threads(2)
    output=ROOT/'diagnostics/separated_goal_derivations_v21.json';assert not output.exists()
    checkpoint=ROOT/'run_separated_goal_v21/checkpoint_001000.pt'
    assert sha(checkpoint)==json.loads(checkpoint.with_suffix('.COMMITTED.json').read_text())['sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);manifest=saved['manifest']
    model=RawBoundaryWorld(Config(**saved['model_config'])).cuda().eval();model.load_state_dict(saved['model'])
    features=RawBoundaryFeatures(ROOT/'prequant_visual_features_v18','cuda');folder=Path(manifest['data']['source'])
    windows,_=load_windows(folder,manifest['data']['train_layouts'])
    source_rows=json.loads((folder/'training_rows.json').read_text());codes=np.load(folder/'visual_tokens.npy');by_episode=defaultdict(list)
    for i,row in enumerate(source_rows):by_episode[row['episode_id']].append(i)
    selected=random.Random(744).sample([w for w in windows if w['start']==0 and (w['supervised']|~w['action_valid']).all() and w['goal_index']>=4],128)
    rows=[];counts=Counter()
    for w in selected:
        d=batch([w],'cuda');raw,_=features.demonstrations([w])
        generated=model.generate(d['task'],d['history'][:,:1],d['previous_actions'][:,:1],d['history_length'],start=0,boundary_features=raw)
        observed=model.training_targets(generated,d['targets'],d['actions']).argmax(-1)[0,:,0].cpu().numpy()
        initial=np.array([observed[0]]);program=export_program(generated)
        # The exported model program is complete before any final-frame label is computed.
        full=torch.tensor(codes[by_episode[w['episode_id']]],device='cuda').long()[None]
        terminal=model.encode_sequence(generated,full)[0][0,-1].argmax().reshape(1).cpu().numpy()
        old=plan(program,initial);new=plan(program,initial,terminal_goal=terminal)
        n=int(w['goal_index']);teacher=[int(x) for x in w['actions'][:n]];table=program.transitions.argmax(-1)
        state=int(initial[0]);correct=0;one=0
        for t,a in enumerate(teacher):
            one+=int(table[observed[t],a]==observed[t+1]);state=int(table[state,a]);correct+=int(state==observed[t+1])
            counts['observed_STOP_identity_correct']+=int(table[observed[t],0]==observed[t]);counts['observed_states']+=1
        counts['teacher_actions']+=n;counts['autonomous_teacher_successor_correct']+=correct;counts['one_step_teacher_successor_correct']+=one
        counts['generated_subgoal_matches_demonstration']+=int(program.subgoal[0]==observed[n])
        counts['all_table_STOP_identity_correct']+=int((table[:,0]==np.arange(len(table))).sum());counts['all_table_states']+=len(table)
        rows.append(dict(episode=w['episode_id'],teacher=teacher,old_plan=old['actions'],diagnostic_plan=new['actions'],
            subgoal=program.subgoal.tolist(),actual_terminal_code=terminal.tolist(),actual_subgoal=int(observed[n]),initial=initial.tolist(),teacher_terminal=0 in teacher))
    def stats(key):
        return dict(first_action_correct=sum(bool(r[key]) and r[key][0]==r['teacher'][0] for r in rows),
            complete_teacher_sequence_correct=sum(r[key]==r['teacher'] for r in rows),no_plan=sum(not r[key] for r in rows),
            nonterminal_STOP=sum(not r['teacher_terminal'] and bool(r[key]) and r[key][-1]==0 for r in rows),
            terminal_STOP=sum(r['teacher_terminal'] and bool(r[key]) and r[key][-1]==0 for r in rows))
    result=dict(checkpoint=str(checkpoint),sha256=sha(checkpoint),training_only=True,oracle_intervention_not_deployable=True,
        samples=len(rows),nonterminal_windows=sum(not r['teacher_terminal'] for r in rows),old=stats('old_plan'),
        task_goal_intervention=stats('diagnostic_plan'),dynamics=dict(counts),rows=rows)
    output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
