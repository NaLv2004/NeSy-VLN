"""Frozen tables, same94 teacher roots and same24 training routes as v49/v46."""
import importlib.util,json,os,random,sys,time,fcntl
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
sys.path.insert(1,str(ROOT/'src_boundary_balanced_v43'))
from model import FactoredWorld,Config,hard
from data import sha
from planner import Program,plan
from evaluate import evaluate,summarize,atomic
spec=importlib.util.spec_from_file_location('legacy_probability',ROOT/'src_boundary_balanced_v43/planner.py')
legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)


@torch.no_grad()
def roots(model,saved,out):
    dataset=json.loads((ROOT/'diagnostic_dinov2_v43/dataset.json').read_text())
    names=random.Random(491017).sample(dataset['fit_episode_ids'],48)
    rows=json.loads((Path(saved['manifest']['data']['source'])/'training_rows.json').read_text())
    raw=np.load(ROOT/'prequant_visual_features_v18/features.npy',mmap_mode='r')
    groups={name:[] for name in names}
    for i,row in enumerate(rows):
        if row['episode_id'] in groups:groups[row['episode_id']].append(i)
    prior=json.loads((ROOT/'diagnostic_path_likelihood_v49/teacher_root_report.json').read_text())
    prior={(r['episode'],r['start']):r for r in prior['records']}
    records=[]
    for name in names:
        indices=groups[name];n=len(indices)
        x=torch.tensor(raw[indices].copy(),device='cuda')[None]
        previous=torch.tensor([[17]+[rows[i]['label'] for i in indices[:-1]]],device='cuda')
        task=torch.tensor([['chair','table'].index(rows[indices[0]]['category'])],device='cuda')
        logits,z,_=model.observation_encoding(x);o=hard(logits)
        memory,_,_=model.memory_sequence(o,previous,z);contexts=model.context_sequence(x,o,previous,task)
        for start in sorted({0,min(n-1,max(1,n//2))}):
            horizon=32-start
            if horizon<1:continue
            table=model.compile(contexts[:,start],task)
            s=int(memory[0,start].argmax());ob=int(logits[0,start].argmax());label=int(rows[indices[start]]['label'])
            old=legacy.Program(**table).validate();new=Program(**table).validate()
            a=legacy.plan(old,s,ob,horizon);b=plan(new,s,ob,horizon)
            previous=prior[(name,start)]
            assert previous['teacher_action']==label
            assert previous['expected_action']==(a['actions'][0] if a['found'] else None)
            assert old.digest()==new.digest()
            records.append(dict(episode=name,start=start,teacher_action=label,
                legacy_action=a['actions'][0] if a['found'] else None,
                stable_action=b['actions'][0] if b['found'] else None,
                legacy_probability=a['model_goal_probability'],stable_probability=b['model_goal_probability'],
                stable_failure_probability=b['model_failure_probability'],
                maximum_row_mass_error=float(np.abs(table['outcome_probability'].astype(np.float64).sum(-1)-1).max()),
                table_digest=new.digest(),unchanged_generated_goal=table['goal']))
    def summary(rs):
        return dict(examples=len(rs),legacy_teacher_agreement=float(np.mean([r['legacy_action']==r['teacher_action'] for r in rs])),
            stable_teacher_agreement=float(np.mean([r['stable_action']==r['teacher_action'] for r in rs])),
            changed_fraction=float(np.mean([r['legacy_action']!=r['stable_action'] for r in rs])),
            stable_stop_fraction=float(np.mean([r['stable_action']==0 for r in rs])))
    report=dict(summary=summary(records),nonstop_teacher=summary([r for r in records if r['teacher_action']!=0]),
        initial_roots=summary([r for r in records if r['start']==0]),records=records,training_only=True,
        limitation='Teacher agreement is not physical navigation success; no model parameters or rules changed.')
    atomic(out/'teacher_roots.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.20)
    out=ROOT/'diagnostic_stable_probability_v50';out.mkdir(exist_ok=True)
    lock=(out/'job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'report.json').exists(),'Completed immutable diagnostic'
    source=ROOT/'diagnostic_sparse_support_v44';commit=json.loads((source/'source.COMMITTED.json').read_text())
    assert sha(source/'source.pt')==commit['sha256']
    saved=torch.load(source/'source.pt',map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).cuda().eval();model.load_state_dict(saved['model']);model.requires_grad_(False)
    began=time.time();atomic(out/'status.json',dict(pid=os.getpid(),state='running',phase='94_teacher_roots',time=time.time()))
    if not (out/'teacher_roots.json').exists(): roots(model,saved,out)
    # Match the already-completed v46 physical baseline exactly, including TF32.
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    baseline_root=ROOT/'diagnostic_discounted_rules_v46'
    original=json.loads((baseline_root/'manifest.json').read_text())
    assert original['source_commit']['sha256']==commit['sha256']
    tasks=original['episodes'];assert len(tasks)==24
    atomic(out/'manifest.json',dict(source_commit=commit,episodes=tasks,training_only=True,max_actions=32,
        baseline_reused=str(baseline_root/'baseline'),no_baseline_rerun=True,arithmetic='normalized_float64_dual_success_failure_v50'))
    atomic(out/'status.json',dict(pid=os.getpid(),state='running',phase='24_matched_training_routes',time=time.time()))
    physical=out/'physical'
    if (physical/'summary.json').exists(): result=json.loads((physical/'summary.json').read_text())
    else:
        rows=evaluate(model,saved['manifest'],physical,workers=4,episodes_override=tasks)
        assert len(rows)==24;result=summarize(rows);atomic(physical/'summary.json',result)
    baseline=json.loads((baseline_root/'baseline/summary.json').read_text())
    report=dict(time=time.time(),source_commit=commit,training_only=True,seconds=time.time()-began,
        baseline=baseline,stable=result,arithmetic_contract=json.loads((Path(__file__).parent/'arithmetic_contract.json').read_text()),
        root_summary=json.loads((out/'teacher_roots.json').read_text())['summary'],
        no_model_training=True,no_baseline_rerun=True,no_finalfixed80_used=True)
    atomic(out/'report.json',report);atomic(out/'status.json',dict(pid=os.getpid(),state='complete',time=time.time()))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
