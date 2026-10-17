"""Resume v8 from its intact optimizer/RNG checkpoint after a verified FP16 fix."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
original=root/'run_rollout_proof_v8';src=root/'src_rollout_proof_v8r1';out=root/'run_rollout_proof_v8r1'
if out.exists() and (out/'launcher.json').exists():raise RuntimeError('Repair already launched; inspect its handle')
if out.exists() and set(p.name for p in out.iterdir())-{'latest.pt','latest.COMMITTED.json'}:
    raise RuntimeError('Unrecognized partial preparation; inspect before recovery')
old=json.loads((original/'launcher.json').read_text())
for folder in [original,root/'gradient_replay_v8']:
    launch=json.loads((folder/'launcher.json').read_text())
    if (Path('/proc')/str(launch['pid'])).exists():raise RuntimeError('An original/diagnostic handle is still present')
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.splitlines()}
if any(memory[g]>300 for g in [3,4,5,7]):raise RuntimeError('Authorized cards busy; no process is stopped')
commit=json.loads((original/'latest.COMMITTED.json').read_text())
assert commit['update']==1400 and hashlib.sha256((original/'latest.pt').read_bytes()).hexdigest()==commit['sha256']
evidence=json.loads((root/'gradient_replay_v8/backward_precision.json').read_text())
assert evidence[0]['amp'] and evidence[0]['module_stats']['history:0']['forward']['dtype']=='torch.float32'
assert all(v['finite'] for v in evidence[0]['parameter_stats'].values())
out.mkdir(exist_ok=True)
for name in ['latest.pt','latest.COMMITTED.json']:shutil.copyfile(original/name,out/name)
manifest=json.loads((original/'manifest.json').read_text());manifest.update(
    precision_repair='Boundary history GRU explicitly FP32; generated rules retain BF16 AMP',
    repair_source_sha256=hashlib.sha256((src/'predictive_nesy/graph_subgoal_model.py').read_bytes()).hexdigest(),
    resume_origin=dict(absolute_path=str(original/'latest.pt'),**commit),repair_preflight=str(root/'gradient_replay_v8/backward_precision.json'))
cmd=old['command'].copy();cmd[cmd.index(str(root/'src_rollout_proof_v8/train_rollout_proof.py'))]=str(src/'train_rollout_proof.py')
cmd[cmd.index('--out')+1]=str(out);cmd+=['--resume']
manifest['train_args']['out']=str(out);manifest['train_args']['resume']=True
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3,4,5',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(src),NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
log=(out/'driver.log').open('ab',buffering=0);process=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=process.pid,time=time.time(),source=str(src),command=cmd,train_gpus=[3,4,5],evaluation_gpu=7,restored_commit=commit)
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
