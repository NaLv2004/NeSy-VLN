"""Immutable training-only comparison on the same exported probability tables."""
import importlib.util,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
sys.path.insert(1,str(ROOT/'src_boundary_balanced_v43'))
from model import FactoredWorld,Config,hard
from data import sha
from planner import Program,plan
spec=importlib.util.spec_from_file_location('expected_probability_planner',ROOT/'src_boundary_balanced_v43/planner.py')
expected=importlib.util.module_from_spec(spec);sys.modules[spec.name]=expected;spec.loader.exec_module(expected)

@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.2)
    out=ROOT/'diagnostic_path_likelihood_v49';out.mkdir(exist_ok=True)
    assert not (out/'teacher_root_report.json').exists(),'Completed immutable audit'
    source=ROOT/'diagnostic_sparse_support_v44';commit=json.loads((source/'source.COMMITTED.json').read_text())
    assert sha(source/'source.pt')==commit['sha256']
    saved=torch.load(source/'source.pt',map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).cuda().eval();model.load_state_dict(saved['model'])
    dataset=json.loads((ROOT/'diagnostic_dinov2_v43/dataset.json').read_text())
    names=random.Random(491017).sample(dataset['fit_episode_ids'],48)
    rows=json.loads((Path(saved['manifest']['data']['source'])/'training_rows.json').read_text())
    raw=np.load(ROOT/'prequant_visual_features_v18/features.npy',mmap_mode='r')
    groups={name:[] for name in names}
    for i,row in enumerate(rows):
        if row['episode_id'] in groups:groups[row['episode_id']].append(i)
    began=time.time();records=[]
    for name in names:
        indices=groups[name];n=len(indices)
        assert n<=33 and rows[indices[-1]]['label']==0
        x=torch.tensor(raw[indices].copy(),device='cuda')[None]
        previous=torch.tensor([[17]+[rows[i]['label'] for i in indices[:-1]]],device='cuda')
        task=torch.tensor([['chair','table'].index(rows[indices[0]]['category'])],device='cuda')
        logits,z,_=model.observation_encoding(x);o=hard(logits)
        memory,_,_=model.memory_sequence(o,previous,z);contexts=model.context_sequence(x,o,previous,task)
        for start in sorted({0,min(n-1,max(1,n//2))}):
            context=contexts[:,start];table=model.compile(context,task)
            s=int(memory[0,start].argmax());ob=int(logits[0,start].argmax());label=int(rows[indices[start]]['label'])
            horizon=32-start
            if horizon<1:continue
            ep=expected.Program(**table).validate();mp=Program(**table).validate()
            a=expected.plan(ep,s,ob,horizon);b=plan(mp,s,ob,horizon)
            current=s*64+ob;next_ids=table['next_memory'][current]
            continuation=ep._levels[horizon-1].reshape(32,64)[next_ids]
            q=(table['outcome_probability'][current]*continuation).sum(-1)
            gm,go=divmod(table['goal'],64);q[0]=float(next_ids[0]==gm)*table['outcome_probability'][current,0,go]
            ordered=np.sort(q)
            record=dict(episode=name,start=start,teacher_action=label,
                expected_action=a['actions'][0] if a['found'] else None,
                path_action=b['actions'][0] if b['found'] else None,
                expected_score=a['model_goal_probability'],path_probability=b['maximum_path_probability'],
                expected_margin=float(ordered[-1]-ordered[-2]),expected_action_range=float(np.ptp(q)),
                max_path_length=len(b['actions']),expected_witness_length=len(a['actions']),
                unchanged_goal=table['goal'])
            records.append(record)
    summary=dict(examples=len(records),expected_action_matches_teacher=float(np.mean([r['expected_action']==r['teacher_action'] for r in records])),
        path_action_matches_teacher=float(np.mean([r['path_action']==r['teacher_action'] for r in records])),
        action_changed_fraction=float(np.mean([r['expected_action']!=r['path_action'] for r in records])),
        median_expected_score=float(np.median([r['expected_score'] for r in records])),
        median_path_probability=float(np.median([r['path_probability'] for r in records])),
        median_top_action_margin=float(np.median([r['expected_margin'] for r in records])),
        expected_margin_under_1e_6=float(np.mean([r['expected_margin']<1e-6 for r in records])),
        median_max_path_length=float(np.median([r['max_path_length'] for r in records])))
    report=dict(training_only=True,source_commit=commit,summary=summary,records=records,seconds=time.time()-began,
        no_environment_rollouts=True,no_model_modified=True,
        limitation='Teacher-action agreement is not navigation success; alternative valid actions may differ.')
    (out/'teacher_root_report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
if __name__=='__main__':main()
