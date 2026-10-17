import argparse,json,os,time
from pathlib import Path
import torch
from model import FactoredWorld,Config
from evaluate import evaluate,atomic
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
OUT=ROOT/'diagnostic_ensemble_rules_v61'
parser=argparse.ArgumentParser();parser.add_argument('--rank',type=int,required=True);a=parser.parse_args()
assert a.rank in (0,1) and os.environ.get('CUDA_VISIBLE_DEVICES')==str(6+a.rank)
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
source=torch.load(ROOT/'onpolicy_successors_v60/source.pt',map_location='cpu',weights_only=False)
heads=torch.load(OUT/'heads.pt',map_location='cpu',weights_only=False)
model=FactoredWorld(Config(**source['model_config'])).cuda().eval().requires_grad_(False)
model.load_candidate(source['model'],heads['heads'])
assert heads['source']['sha256']=='58abc7f6a46587f2312412b1848c1e84dc64a374ba725b81f51dd8c0bca880bf'
manifest=dict(source['manifest']);manifest.update(architecture='ensemble_rules_v61',runtime_planner='NumPy max/min reachability',
    base_checkpoint=heads['source'],candidate_head_selection_used_dev80=False,baseline_source_selected_by_dev80=True)
atomic(OUT/f'eval_rank{a.rank}.status.json',dict(state='evaluating',pid=os.getpid(),time=time.time()))
rows=evaluate(model,manifest,OUT/'dev_full80',rank=a.rank,world_size=2,workers=8)
assert len(rows)==40
atomic(OUT/f'eval_rank{a.rank}.status.json',dict(state='complete',pid=os.getpid(),time=time.time(),episodes=40,successes=sum(r['success'] for r in rows)))
