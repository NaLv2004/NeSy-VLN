"""Offline empirical aliasing in identical symbolic premises; no policy run."""
import collections,json,os,time
import torch
from model import FactoredWorld,Config
from data import ROOT,sha
from joint_data import JointData
from history_data import HistoryData
from accelerated_transition import install

torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=True
assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
began=time.time();source=ROOT/'run_goal_probability_v68';commit=json.loads((source/'best.COMMITTED.json').read_text())
path=source/commit['filename'];assert sha(path)==commit['sha256']
saved=torch.load(path,map_location='cpu',weights_only=False)
model=FactoredWorld(Config(**saved['model_config'])).cuda().eval().requires_grad_(False);model.load_state_dict(saved['model']);install(model)
data=JointData(model,saved['manifest']['transition_holdout_episodes'],torch.device('cuda'));hist=HistoryData(model,data)
x={k:v.cpu().numpy() for k,v in hist.columns.items()}
def analyze(indices,length):
    groups=collections.defaultdict(collections.Counter)
    for i in indices:
        if x['action'][i]==0:continue
        key=(int(x['program'][i]),int(x['memory'][i]),int(x['action'][i]),*x['obs'][i,-length:].tolist())
        if length>1:key+=tuple(x['previous'][i,1-length:].tolist())
        groups[key][int(x['target'][i])]+=1
    repeated=[c for c in groups.values() if sum(c.values())>=2]
    total=sum(sum(c.values()) for c in repeated);errors=sum(sum(c.values())-max(c.values()) for c in repeated)
    return dict(total_nonstop_rows=sum(sum(c.values()) for c in groups.values()),groups=len(groups),
        repeated_groups=len(repeated),rows_in_repeated_groups=total,contradictory_groups=sum(len(c)>1 for c in repeated),
        empirical_best_one_prediction_accuracy_on_repeated=1-errors/max(1,total),
        unavoidable_disagreements_for_identical_inputs=errors,
        caveat='Empirical identity collisions only, not heldout generalization ceiling; longer histories reduce repeated coverage')
report=dict(time=time.time(),source=commit,seconds=time.time()-began,optimizer_steps=0,new_rollouts=0,development80_used=False,
    groups={part:{str(k):analyze(ids.cpu().tolist(),k) for k in (1,2,4,8)} for part,ids in [('fit',hist.fit),('holdout',hist.holdout)]})
out=ROOT/'run_history_transition_v69/alias_audit.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
