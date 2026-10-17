"""Diagnose immutable completed navigation traces, without a new rollout.

Inspect whether the unchanged generated joint goal has any STOP preimage,
and whether an observed goal-code was blocked by predictive memory. Runtime
rules/goals are never changed by this audit. No semantic state labels.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
sys.path.insert(0,str(ROOT/'src_goal_equivalence_v57'))
from model import FactoredWorld,Config
from data import sha


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='run_definition_rules_v56')
    parser.add_argument('--update',type=int,default=2000)
    args=parser.parse_args()
    assert args.run in ('run_definition_rules_v56','run_goal_equivalence_v57')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(2);os.nice(10);began=time.time()
    directory=ROOT/args.run/f'dev_{args.update:06d}'
    summary=json.loads((directory/'summary.json').read_text())
    assert summary['episodes']==80
    files=sorted(directory.glob('*/trace.json'));assert len(files)==80
    models={};episodes=[];programs=[];steps=[]
    for path in files:
        record=json.loads(path.read_text());tables={}
        for program in record['programs']:
            with np.load(path.parent/program['file'],allow_pickle=False) as recipe:
                model_path=directory/str(recipe['model_file'])
                if str(model_path) not in models:
                    assert sha(model_path)==str(recipe['model_sha256'])
                    saved=torch.load(model_path,map_location='cpu',weights_only=False)
                    model=FactoredWorld(Config(**saved['model_config'])).eval()
                    model.load_state_dict(saved['model'],strict=True)
                    models[str(model_path)]=model
                model=models[str(model_path)];c=model.cfg
                context=torch.from_numpy(recipe['context'].copy())
                gm,go=divmod(program['goal'],c.observations)
                memory=F.one_hot(torch.arange(c.memories)[None],c.memories).float()
                observed=F.one_hot(torch.full((1,c.memories),go),c.observations).float()
                nm,no=model.transition(context,memory,observed,torch.zeros(1,c.memories,dtype=torch.long))
                stop_next_memory=nm.argmax(-1)[0].numpy()
                assert np.all(no.argmax(-1)[0].numpy()==go), 'Learned STOP effect no longer preserves O'
                preimage=bool((stop_next_memory==gm).any())
                tables[program['start']]=dict(goal_memory=gm,goal_observation=go,stop_next_memory=stop_next_memory)
                programs.append(dict(episode=record['episode']['episode_id'],start=program['start'],
                    success=bool(record['metrics'].get('success')),goal_memory=gm,goal_observation=go,
                    stop_preimage_count=int((stop_next_memory==gm).sum()),
                    stored_preimage=program['terminal_preimage'],cpu_preimage=preimage,
                    cpu_matches_stored=preimage==program['terminal_preimage']))
        episode_steps=[]
        for step in record['steps']:
            boundary=max(k for k in tables if k<=step['step']);table=tables[boundary]
            observed_goal=step['observed']==table['goal_observation']
            would_reach=observed_goal and int(table['stop_next_memory'][step['memory']])==table['goal_memory']
            row=dict(episode=record['episode']['episode_id'],step=step['step'],
                success=bool(record['metrics'].get('success')),observed_goal=observed_goal,
                memory_blocks_stop=observed_goal and not would_reach,
                exact_stop_supported=would_reach,action=step['action'],renewed=step['renewed'])
            steps.append(row);episode_steps.append(row)
        eps_programs=[p for p in programs if p['episode']==record['episode']['episode_id']]
        episodes.append(dict(episode=record['episode']['episode_id'],success=bool(record['metrics'].get('success')),
            any_no_preimage=any(not p['stored_preimage'] for p in eps_programs),
            every_program_no_preimage=bool(eps_programs) and all(not p['stored_preimage'] for p in eps_programs),
            observed_goal_any=any(s['observed_goal'] for s in episode_steps),
            memory_blocked_any=any(s['memory_blocks_stop'] for s in episode_steps),
            exact_stop_supported_any=any(s['exact_stop_supported'] for s in episode_steps),
            planning_failed=record.get('planning_failed',False),
            failed_horizon=record.get('failed_proof',{}).get('horizon'),
            final_distance=record['metrics'].get('distance_to_goal')))
    def stats(values,fields):return {f:sum(bool(r[f]) for r in values) for f in fields}
    result=dict(time=time.time(),seconds=time.time()-began,run=args.run,update=args.update,
        original_summary=summary,programs=len(programs),steps=len(steps),
        program_counts=stats(programs,['stored_preimage','cpu_preimage','cpu_matches_stored']),
        episode_counts={name:dict(episodes=len(group),**stats(group,['any_no_preimage','every_program_no_preimage',
            'observed_goal_any','memory_blocked_any','exact_stop_supported_any','planning_failed']))
            for name,group in [('all',episodes),('failed',[r for r in episodes if not r['success']])]},
        step_counts=stats(steps,['observed_goal','memory_blocks_stop','exact_stop_supported']),
        generated_goal_memory_histogram=dict(Counter(p['goal_memory'] for p in programs)),
        episode_rows=episodes,program_rows=programs,step_rows=steps,
        new_rollouts=0,optimizer_steps=0,finalfixed80_used=False,live_training_changed=False,
        caveat='Observed goal-code equality is not physical success. CPU STOP preimage agrees/disagrees with the archived GPU boolean as reported; no table is substituted at runtime.')
    out=ROOT/'diagnostic_probability_training_v58'/f'goal_feasibility_{args.run}_{args.update}.json'
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('_rows') and k!='original_summary'}),flush=True)

if __name__=='__main__':main()
