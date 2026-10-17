import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from nesy_vla.runtime import read,setup,DataBank,load_checkpoint,atomic
from nesy_vla.model import NavigationModel
from nesy_vla.evaluation import evaluate_tasks,aggregate

p=argparse.ArgumentParser();p.add_argument('--config',default='configs/navigation.json')
p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
p.add_argument('--suite',choices=['fixed80','development'],default='fixed80')
p.add_argument('--stochastic',action='store_true');p.add_argument('--no-revision',action='store_true')
p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1);p.add_argument('--gpu',type=int,default=0)
a=p.parse_args();cfg=read(a.config);setup(cfg);torch.cuda.set_device(a.gpu);bank=DataBank(cfg)
model=NavigationModel(cfg).cuda().eval();state=load_checkpoint(a.checkpoint,model)
torch.manual_seed(cfg['seed']+a.rank);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
sim=bank.simulator(out/f'sim_{a.rank}.log',a.gpu,True)
try:
    tasks=bank.fixed if a.suite=='fixed80' else bank.dev
    rows=evaluate_tasks(model,sim,tasks[a.rank::a.world],out,a.stochastic,not a.no_revision)
    atomic(out/f'rank{a.rank}.json',rows)
    atomic(out/f'summary_rank{a.rank}.json',dict(aggregate(rows),checkpoint=a.checkpoint,update=state['update']))
finally:sim.close()
