import json,os,time
from pathlib import Path
import torch
from model import FactoredWorld,Config
from planner import Program,plan,audit
from program_archive import snapshot,save_recipe,restore,sha

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='7'
torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=True
run=ROOT/'run_boundary_balanced_v43';commit=json.loads((run/'latest.COMMITTED.json').read_text())
assert commit['update']==4500 and sha(run/commit['filename'])==commit['sha256']
saved=torch.load(run/commit['filename'],map_location='cpu',weights_only=False)
model=FactoredWorld(Config(**saved['model_config'])).cuda().eval();model.load_state_dict(saved['model'])
out=ROOT/'diagnostic_program_archive_v43';out.mkdir(exist_ok=True)
archive=snapshot(model,out,0);checks=[]
torch.manual_seed(431017)
for index in range(3):
    context=torch.randn(1,model.cfg.context,device='cuda');task=torch.tensor([index%2],device='cuda')
    original=Program(**model.compile(context,task)).validate()
    folder=out/f'case{index}';folder.mkdir(exist_ok=True)
    file=folder/'recipe_00.npz';save_recipe(file,context,task,original,archive)
    reloaded=restore(file)
    proof=plan(original,index,index*3,horizon=16);replay=plan(reloaded,index,index*3,horizon=16)
    assert proof==replay
    if proof['found']:audit(reloaded,proof)
    checks.append(dict(digest=original.digest(),recipe_bytes=file.stat().st_size,proof_identical=True))
record=dict(time=time.time(),checkpoint=commit,checks=checks,model_bytes=(out/archive['file']).stat().st_size,
    full_table_bytes=original.next_memory.nbytes+original.outcome_probability.nbytes,changes_policy=False)
(out/'report.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
