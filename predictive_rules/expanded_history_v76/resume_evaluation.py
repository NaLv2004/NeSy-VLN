"""Evaluate the sealed predictor after repairing the batching loop; no training."""
import os,json,time
import torch
import torch.distributed as dist
from model import FactoredWorld,Config
from data import ROOT,sha
from history_model import HistoryTransition
from accelerated_transition import install
from evaluate import evaluate,summarize,atomic

def main():
    local=int(os.environ['LOCAL_RANK']);rank=int(os.environ['RANK'])
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5'
    torch.cuda.set_device(local);torch.set_num_threads(3);dist.init_process_group('nccl')
    out=ROOT/'run_expanded_history_v76'
    manifest=json.loads((out/'manifest.json').read_text())
    origin=manifest['history_source_checkpoint'];path=origin['path'];assert sha(path)==origin['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=FactoredWorld(Config(**saved['model_config'])).cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['model']);install(model);del saved
    selected=json.loads((out/'best.COMMITTED.json').read_text());path=out/selected['filename'];assert sha(path)==selected['sha256']
    predictor=HistoryTransition().cuda().eval().requires_grad_(False)
    predictor.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['predictor'])
    atomic(out/f'status_rank{rank}.json',dict(state='evaluating_repaired',pid=os.getpid(),update=selected['update'],time=time.time()))
    rows=evaluate(model,manifest,out/'development_best_repaired',rank,2,workers=40,history_predictor=predictor)
    all_rows=[None,None];dist.all_gather_object(all_rows,rows)
    if rank==0:
        summary=summarize([r for group in all_rows for r in group])
        atomic(out/'development_best_repaired/summary.json',summary)
        atomic(out/'result.json',dict(state='complete',updates=1000,time=time.time(),chosen=selected,development=summary,
            source_development_successes=44,source_episodes=80,goal_complete=False,finalfixed80_used=False,
            repair='Removed mutation of iteration list; evaluated same sealed predictor without training'))
    atomic(out/f'status_rank{rank}.json',dict(state='complete',pid=os.getpid(),time=time.time(),updates=1000))
    dist.destroy_process_group()

if __name__=='__main__':main()
