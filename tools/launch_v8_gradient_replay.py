"""Replay the committed optimizer/RNG on authorized idle cards, without replacing it."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
source=root/'src_rollout_proof_v8';original=root/'run_rollout_proof_v8'
out=root/'gradient_replay_v8';src=root/'src_gradient_replay_v8'
if out.exists() or src.exists():raise RuntimeError('Diagnostic already exists; inspect its process, never duplicate')
launcher=json.loads((original/'launcher.json').read_text())
if (Path('/proc')/str(launcher['pid'])).exists():raise RuntimeError('Original handle is still present')
raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.splitlines()}
if any(memory[g]>300 for g in [3,4,5]):raise RuntimeError('Authorized cards are busy')
commit=json.loads((original/'latest.COMMITTED.json').read_text())
assert hashlib.sha256((original/'latest.pt').read_bytes()).hexdigest()==commit['sha256'] and commit['update']==1400
shutil.copytree(source,src);out.mkdir()
for name in ['latest.pt','latest.COMMITTED.json','manifest.json']:
    shutil.copyfile(original/name,out/name)
path=src/'train_rollout_proof.py';text=path.read_text()
text=text.replace("begin=time.time();model.train();start=rng.choice(starts);optimizer.zero_grad(set_to_none=True)",
    "begin=time.time();replay_rng=rng.getstate();model.train();start=rng.choice(starts);optimizer.zero_grad(set_to_none=True)")
old="""        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')"""
new="""        gradient_max=max(float(x.grad.detach().abs().max()) for x in model.parameters() if x.grad is not None)
        bad=[n for n,x in model.named_parameters() if x.grad is not None and not torch.isfinite(x.grad).all()]
        diagnostic_norm=torch.stack([x.grad.detach().double().norm() for x in model.parameters() if x.grad is not None]).norm()
        if rank==0:
            with (out/'gradient_audit.jsonl').open('a') as stream:stream.write(json.dumps(dict(attempted_update=step+1,max_abs=gradient_max,bad_parameters=bad,norm_fp64=float(diagnostic_norm)))+'\\n')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),update=step,model_config=model.export_config(),manifest=manifest,
                replay_rng=replay_rng,rank=rank,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),out/f'failure_rank{rank}.pt')
            atomic(out/f'failure_rank{rank}.json',dict(attempted_update=step+1,max_abs=gradient_max,bad_parameters=bad,norm_fp64=float(diagnostic_norm),norm_fp32=float(norm)))
            raise FloatingPointError('Captured nonfinite gradient diagnostic')"""
assert old in text;text=text.replace(old,new);path.write_text(text)
cmd=launcher['command'].copy();cmd[cmd.index(str(source/'train_rollout_proof.py'))]=str(path)
cmd[cmd.index('--out')+1]=str(out);cmd[cmd.index('--updates')+1]='1420';cmd+=['--resume']
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3,4,5',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(src),NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
log=(out/'driver.log').open('ab',buffering=0)
process=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=process.pid,time=time.time(),source=str(src),command=cmd,original_commit=commit)
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
