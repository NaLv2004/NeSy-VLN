"""Replay saved discrete programs using the exact inference-time network weights.

No simulator or visual model is loaded. Counterfactuals operate on symbol tokens
and generated rules. Reported state graphs contain no spatial coordinates.
"""
import argparse,json,sys,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nesy_vla.program import Program
from nesy_vla.symbols import tokenize
from nesy_vla.neural_logic import NeuralSymbolicLogic
from nesy_vla.logic import NoSupportedAction


@torch.no_grad()
def audit(folder):
    folder=Path(folder);trajectory=json.loads((folder/'trajectory.json').read_text())
    programs={}
    for file in folder.glob('program_*.json'):
        raw=json.loads(file.read_text())['program']
        p=Program.from_record(raw)
        programs[p.version]=p
    p=next(iter(programs.values()));k=len(p.definitions)
    logic=NeuralSymbolicLogic(k,p.action_heads.shape[-1],len(p.weights))
    logic.load_state_dict(torch.load(folder/'logic_snapshot.pt',map_location='cpu',weights_only=True))
    logic.eval();rows=trajectory['steps'];fixed={};errors=[];changes=collections.Counter();states=set();edges=[]
    choices=[];expert=correct=0;counterfactual_tv=[]
    for index,row in enumerate(rows):
        p=programs[row['version']];bits=torch.tensor(row['symbol_codes'],dtype=torch.float)
        state=tokenize(bits);observed=tokenize(torch.tensor(row['perceptual_codes'],dtype=torch.float))
        prior=None if row['prior_codes'] is None else tokenize(torch.tensor(row['prior_codes'],dtype=torch.float))
        replay_state=logic.assimilate(observed,prior)
        if not torch.equal(replay_state.codes,state.codes):raise AssertionError('Saved symbolic state update cannot be replayed')
        result=logic.infer(p,state);prob=result['probs'];choice=int(prob.argmax());choices.append(choice)
        errors.append(float((result['support']-torch.tensor(row['support'])).abs().max()))
        errors.append(float((result['action_contribution']-torch.tensor(row['contribution'])).abs().max()))
        first=fixed.setdefault(row['version'],state)
        alternatives={'first_state':first,'flipped_state':tokenize(1-bits),'no_symbolic_memory':observed}
        for name,other in alternatives.items():
            alternate=logic.infer(p,other)['probs']
            changes[name]+=int(alternate.argmax()!=choice)
            if name=='first_state':counterfactual_tv.append(float((alternate-prob).abs().sum()/2))
        strongest=int(result['action_contribution'][:,choice].argmax())+logic.action_slice.start
        try:changes['remove_strongest_rule']+=int(logic.infer(p,state,disabled_rule=strongest)['probs'].argmax()!=choice)
        except NoSupportedAction:changes['remove_strongest_rule']+=1
        if row.get('expert'):expert+=1;correct+=int(choice==row['action'])
        key=f"v{row['version']}:"+''.join(map(str,row['symbol_codes']));states.add(key)
        if index+1<len(rows):
            after=rows[index+1];destination=f"v{after['version']}:"+''.join(map(str,after['symbol_codes']))
            edges.append({'from':key,'action':row['action'],'to':destination,'strongest_rule':strongest})
    return {'episode_id':trajectory['episode_id'],'success':trajectory['metrics'].get('success'),
        'steps':len(rows),'distinct_symbol_states':len(states),'max_proof_error':max(errors,default=0.),
        'counterfactual_action_changes':dict(changes),'greedy_actions':choices,
        'first_state_probability_tv':sum(counterfactual_tv)/max(1,len(counterfactual_tv)),
        'expert_steps':expert,'expert_correct':correct,'state_transitions':edges,
        'snapshot':str(folder/'logic_snapshot.pt')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',action='append',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args();torch.set_num_threads(2)
    rows=[]
    for folder in args.folder:
        path=Path(folder)
        paths=[path] if (path/'trajectory.json').exists() else [p.parent for p in sorted(path.glob('*/trajectory.json'))]
        rows.extend(audit(p) for p in paths)
    counter=collections.Counter()
    for row in rows:counter.update(row['counterfactual_action_changes'])
    summary={'episodes':len(rows),'steps':sum(r['steps'] for r in rows),
        'max_proof_error':max((r['max_proof_error'] for r in rows),default=None),
        'counterfactual_action_changes':dict(counter),'expert_steps':sum(r['expert_steps'] for r in rows),
        'expert_correct':sum(r['expert_correct'] for r in rows)}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'summary':summary,'episodes':rows},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
