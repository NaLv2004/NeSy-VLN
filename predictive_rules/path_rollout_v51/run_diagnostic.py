"""Same24 training routes, no learning and no rerunning the saved baseline."""
import json,os,sys,time,fcntl
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
sys.path.insert(1,str(ROOT/'src_boundary_balanced_v43'))
import torch
from model import FactoredWorld,Config
from data import sha
from evaluate import evaluate,summarize,atomic

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.20)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    out=ROOT/'diagnostic_path_rollout_v51';out.mkdir(exist_ok=True)
    lock=(out/'job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'report.json').exists()
    source=ROOT/'diagnostic_sparse_support_v44';commit=json.loads((source/'source.COMMITTED.json').read_text())
    assert sha(source/'source.pt')==commit['sha256']
    saved=torch.load(source/'source.pt',map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).cuda().eval();model.load_state_dict(saved['model']);model.requires_grad_(False)
    baseline_root=ROOT/'diagnostic_discounted_rules_v46'
    original=json.loads((baseline_root/'manifest.json').read_text());assert original['source_commit']['sha256']==commit['sha256']
    tasks=original['episodes'];assert len(tasks)==24
    atomic(out/'manifest.json',dict(source_commit=commit,episodes=tasks,training_only=True,max_actions=32,
        planner='normalized_float64_shortest_maximum_likelihood_path_v51',baseline_reused=str(baseline_root/'baseline')))
    atomic(out/'status.json',dict(pid=os.getpid(),state='running',time=time.time()))
    began=time.time();rows=evaluate(model,saved['manifest'],out/'physical',workers=4,episodes_override=tasks)
    assert len(rows)==24
    result=summarize(rows);result['proof_kind']='Maximum single-path likelihood over all learned outcomes; exact ties select shortest path'
    report=dict(time=time.time(),seconds=time.time()-began,training_only=True,source_commit=commit,
        baseline=json.loads((baseline_root/'baseline/summary.json').read_text()),maximum_path=result,
        no_model_training=True,no_baseline_rerun=True,no_finalfixed80_used=True)
    atomic(out/'report.json',report);atomic(out/'status.json',dict(pid=os.getpid(),state='complete',time=time.time()))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
