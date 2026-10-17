import json,os,time
from pathlib import Path
import torch,torch.distributed as dist
from model import FactoredWorld,Config
from evaluate import atomic,evaluate,summarize
from data import ROOT,sha
from curriculum import protocol
OUT=ROOT/'run_normal_near_medium_v67'
def main():
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    assert os.environ['CUDA_VISIBLE_DEVICES']=='6,7'
    torch.cuda.set_device(local);torch.set_num_threads(2);dist.init_process_group('nccl')
    folder=OUT/'baseline_v63_dev80'
    if not (folder/'summary.json').exists():
        old=ROOT/'goal_discrimination_v64';commit=json.loads((old/'source.COMMITTED.json').read_text())
        assert sha(old/'source.pt')==commit['sha256']
        saved=torch.load(old/'source.pt',map_location='cpu',weights_only=False)
        model=FactoredWorld(Config(**saved['model_config'])).cuda().eval();model.load_state_dict(saved['model'])
        manifest=dict(saved['manifest'],curriculum=protocol());del saved
        atomic(OUT/f'status_rank{rank}.json',dict(state='baseline_new_curriculum',pid=os.getpid(),time=time.time()))
        rows=evaluate(model,manifest,folder,rank,2,workers=12)
        gathered=[None,None];dist.all_gather_object(gathered,rows)
        if rank==0:atomic(folder/'summary.json',dict(summarize([r for v in gathered for r in v]),curriculum=protocol(),checkpoint=commit))
    dist.barrier();atomic(OUT/f'status_rank{rank}.json',dict(state='baseline_complete',pid=os.getpid(),time=time.time()))
    dist.destroy_process_group()
if __name__=='__main__':main()
