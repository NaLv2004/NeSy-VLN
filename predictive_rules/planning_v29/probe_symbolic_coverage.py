"""Read-only training-window audit of finite training scores vs real proofs."""
import argparse,json,random
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from planning_model import PlanningObservedWorld,Config
from train import ROOT,Windows,prepare
from hard_interpreter import Program,plan
from predictive_nesy.subgoal_data import batch
from predictive_nesy.replayed_boundary_data import ReplayedBoundaryFeatures
from predictive_nesy.selected_row_objectives import frozen_rule_proof_levels


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();torch.set_num_threads(2);rng=random.Random(30291)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=PlanningObservedWorld(Config(**saved['model_config'])).eval();model.load_state_dict(saved['model'])
    data_source=saved['manifest']['data'];corpus=Windows(data_source['source'],data_source['train_layouts'])
    features=ReplayedBoundaryFeatures(ROOT/'prequant_visual_features_v18',ROOT/'replayed_boundary_features_v20','cpu')
    counts=Counter();per_start={}
    for start in [0,4,10,20]:
        local=Counter()
        for _ in range(2):
            rows=[corpus.window(*k) for k in rng.choices(corpus.by_start[start],k=4)];d=batch(rows,'cpu')
            p=prepare(model,d,features,rows);full=model.rows(p);ids=full.argmax(-1)
            observed=model.encode_sequence(p,d['targets'])[0].argmax(-1)
            goals=p['subgoal_logits'].argmax(-1).flatten();h=min(10,32-start)
            levels=frozen_rule_proof_levels(full,torch.nn.functional.one_hot(goals,512).float(),h)
            predicted=observed[:,0].clone();index=torch.arange(4)
            programs=[Program(ids[j].numpy().astype(np.int16),full[j].max(-1).values.numpy(),
                              np.tile(np.arange(512,dtype=np.int16)[None],(512,1)),int(goals[j]),start).validate() for j in range(4)]
            for j in range(4):
                local['windows']+=1
                endpoint=int(observed[j,int(d['goal_index'][j])]);local['goal_matches_recorded_endpoint']+=int(goals[j])==endpoint
            for t in range(h):
                current=model.correction.selected(p['definitions'],p['context_raw'],predicted,observed[:,t]).argmax(-1)
                depth=min(10,32-start-t)
                for j,program in enumerate(programs):
                    if not bool(d['action_valid'][j,t] and d['supervised'][j,t]):continue
                    a=int(d['actions'][j,t]);s=int(current[j])
                    # C has already been applied above. The identity correction
                    # here only lets the unchanged NumPy solver start at s.
                    proof=plan(program,s,s,horizon=depth)
                    finite=int(levels[depth][j,s].argmax(-1))
                    local['teacher_decisions']+=1;local['finite_score_matches_teacher']+=finite==a
                    local['real_proof_exists']+=proof['found']
                    if proof['found']:
                        local['real_proof_matches_teacher']+=proof['actions'][0]==a
                        local['training_runtime_first_action_disagreement']+=finite!=proof['actions'][0]
                    elif finite==a:local['apparent_matches_without_proof']+=1
                    local['one_step_matches_next_observation']+=int(ids[j,s,a])==int(observed[j,t+1])
                predicted=ids[index,current,d['actions'][:,t]]
        counts.update(local);per_start[start]=dict(local)
    total=counts['teacher_decisions'];found=counts['real_proof_exists']
    result=dict(checkpoint=args.checkpoint,training_only=True,optimizer_updates=0,counts=dict(counts),by_start=per_start,
        actual_proof_rate=found/max(total,1),teacher_action_accuracy_with_real_proof=counts['real_proof_matches_teacher']/max(total,1),
        finite_score_teacher_action_accuracy=counts['finite_score_matches_teacher']/max(total,1),
        agreement_conditional_on_found_proof=counts['real_proof_matches_teacher']/max(found,1))
    assert not counts['training_runtime_first_action_disagreement'],result
    Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
