"""Paired, fixed-weight training-only rollout comparison; no learning."""
from collections import defaultdict
import fcntl,json,os,random,sys,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
sys.path.insert(1,str(ROOT/'src_boundary_balanced_v43'))
import torch
from model import FactoredWorld,Config
from data import sha
from evaluate import evaluate,summarize,atomic
from predictive_nesy.environment import make_bank

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
    torch.set_num_threads(3);torch.cuda.set_per_process_memory_fraction(.20)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    out=ROOT/'diagnostic_discounted_rules_v46';out.mkdir(exist_ok=True)
    lock=(out/'job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (out/'report.json').exists(),'Completed controlled rollout must not be repeated'
    source=ROOT/'diagnostic_sparse_support_v44';commit=json.loads((source/'source.COMMITTED.json').read_text())
    assert sha(source/'source.pt')==commit['sha256']
    saved=torch.load(source/'source.pt',map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).cuda().eval();model.load_state_dict(saved['model']);model.requires_grad_(False)
    bank=make_bank();rng=random.Random(461017);groups=defaultdict(list)
    for episode in bank.train:groups[(episode['domain'],episode['category'],episode['distance_group'])].append(episode)
    assert len(groups)==12
    tasks=[]
    for key,group in sorted(groups.items()):tasks.extend(rng.sample(sorted(group,key=lambda e:e['episode_id']),2))
    assert len(tasks)==24 and not {e['scene'] for e in tasks}&{e['scene'] for e in bank.dev}
    provenance=dict(source_commit=commit,training_only=True,episodes=tasks,seed=461017,max_actions=32,
        discounts=[1.,.95],unchanged_model_and_goals=True,all_actions_symbolically_derived=True,
        rationale='Compare exact goal-reaching probability with uniformly discounted goal-reaching probability; no state reward table or handcrafted predicate.')
    atomic(out/'manifest.json',provenance);results={};began=time.time()
    for label,discount in [('baseline',1.),('discount095',.95)]:
        atomic(out/'status.json',dict(state='running',pid=os.getpid(),variant=label,time=time.time()))
        folder=out/label
        if (folder/'summary.json').exists():result=json.loads((folder/'summary.json').read_text())
        else:
            rows=evaluate(model,saved['manifest'],folder,workers=4,episodes_override=tasks,discount=discount)
            assert len(rows)==24
            result=summarize(rows);result.update(discount=discount,training_only=True)
            atomic(folder/'summary.json',result)
        results[label]=result;print(json.dumps(dict(variant=label,summary=result)),flush=True)
    report=dict(time=time.time(),source_commit=commit,training_only=True,results=results,seconds=time.time()-began,
        independent_validation=False,limit='24 matched training tasks; not dev80 or finalfixed80 performance')
    atomic(out/'report.json',report);atomic(out/'status.json',dict(state='complete',pid=os.getpid(),time=time.time()))
    print(json.dumps(dict(complete=True,seconds=report['seconds'])),flush=True)
if __name__=='__main__':main()
