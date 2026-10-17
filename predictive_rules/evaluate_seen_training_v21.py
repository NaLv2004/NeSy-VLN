"""One closed-loop training-task audit; no new model or validation selection."""
import inspect,json,os,random,time
from pathlib import Path
from collections import defaultdict
import torch
import evaluate_separated_goal_v21 as evaluator
from predictive_nesy.separated_goal_world import SeparatedGoalWorld,Config
from predictive_nesy.subgoal_data import load_windows
from predictive_nesy.environment import make_bank
from predictive_nesy.data import sha

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='7';torch.set_num_threads(2)
    root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015');out=root/'diagnostics/seen_training_navigation_v21';assert not out.exists()
    checkpoint=root/'run_separated_goal_v21/checkpoint_001000.pt'
    assert sha(checkpoint)==json.loads(checkpoint.with_suffix('.COMMITTED.json').read_text())['sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);bank=make_bank()
    windows,_=load_windows(saved['manifest']['data']['source'],saved['manifest']['data']['train_layouts'])
    seen={w['episode_id'] for w in windows if w['start']==0 and w['goal_index']>=4 and (w['supervised']|~w['action_valid']).all()}
    groups=defaultdict(list);rng=random.Random(2174)
    for episode in bank.train:
        if episode['episode_id'] in seen:groups[(episode['domain'],episode['distance_group'],episode['category'])].append(episode)
    for values in groups.values():rng.shuffle(values)
    selected=[]
    while len(selected)<32:
        before=len(selected)
        for key in sorted(groups):
            if groups[key] and len(selected)<32:selected.append(groups[key].pop())
        assert len(selected)>before, 'Insufficient distinct seen training tasks'
    bank.training_probe=selected
    # Reuse the measured interpreter, gated evaluator and independent proof
    # audit verbatim; replace only the explicit task selection and suite count.
    code=inspect.getsource(evaluator.evaluate)
    a="episodes=bank.dev if suite=='development' else bank.fixed";assert code.count(a)==1
    code=code.replace(a,'episodes=bank.training_probe')
    a="assert len(episodes)==80 and not set(e['scene'] for e in episodes)&set(bank.provenance['train_layouts'])";assert code.count(a)==1
    code=code.replace(a,"assert len(episodes)==32 and set(e['scene'] for e in episodes)<=set(bank.provenance['train_layouts'])")
    code=code.replace('total=80','total=32');namespace=dict(vars(evaluator))
    exec(compile(code,'<training-only-closed-loop-audit>','exec'),namespace)
    model=SeparatedGoalWorld(Config(**saved['model_config'])).cuda().eval();model.load_state_dict(saved['model'])
    result=namespace['evaluate'](model,saved['manifest'],bank,out,suite='training_seen32')
    result.update(training_only=True,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),selection_seed=2174,
        task_scope='32 distinct ordinary training tasks with successful supervised starts; stratified across available domain/distance/category groups',
        not_validation_or_generalization=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
