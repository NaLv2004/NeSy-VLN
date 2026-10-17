import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/calibrated_vision_v10')
old=Path('/home/user/xjy/nesy_vla_20261010/symbolic_vision_v8')
assert not (root/'pilot_processes.json').exists()
record=json.loads((root/'calibration_process.json').read_text())
assert not Path(f"/proc/{record['pid']}/cmdline").exists()
proof=json.loads((root/'calibration_provenance.json').read_text())
assert proof['training_only'] and proof['label_free'] and '59 passed' in (root/'verification.log').read_text()
cfg=json.loads((old/'configs/frozen_qwen.json').read_text())
cfg.update(root=str(root),updates=10,demonstrations_per_rank=48,training_chunk_batch=24,maximum_chunk_batch=24,
    training_frame_budget=192,revision_chunk_limit=24,canonical_vision_batch=8,evaluation_every=10,
    vision_cache_entries=2048)
(root/'configs').mkdir(exist_ok=True)
python='/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python';processes=[]
for name,gpu,calibrate in [('uncalibrated_control',6,False),('calibrated_qwen',7,True)]:
    arm=dict(cfg,architecture=name+'_v10')
    if calibrate:arm['vision_calibration']=str(root/'training_feature_calibration.pt')
    (root/'configs'/f'{name}.json').write_text(json.dumps(arm,indent=2))
    out=root/'runs'/name;out.mkdir(parents=True)
    wrapper=root/f'run_{name}.py'
    wrapper.write_text("import runpy,sys,torch\nfrom pathlib import Path\nroot=Path(__file__).resolve().parent\nsys.path.insert(0,str(root))\ntorch.cuda.set_per_process_memory_fraction(.20)\nsys.argv=['scripts/train.py','--config','configs/"+name+".json','--out',str(root/'runs/"+name+"')]\nrunpy.run_path(str(root/'scripts/train.py'),run_name='__main__')\n")
    with (out/'pilot.log').open('ab') as log:
        child=subprocess.Popen([python,'-u',str(wrapper)],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1'))
    processes.append({'name':name,'pid':child.pid,'gpu':gpu,'memory_fraction':.20,'out':str(out)})
manifest={'time':time.time(),'processes':processes,'controlled_difference':'Training-only per-channel centering and scaling of frozen visual features.',
    'matched':['source symbolic model','fresh grounding initialization and optimizer','random seeds','48 demonstrations per update','10 updates','canonical visual batching','fixed split'],
    'all_actions_via_discrete_generated_rules':True,'manual_predicates':False,'maps':False,
    'source_files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for name in ['nesy_vla','scripts','tests','configs'] for p in (root/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts}}
(root/'experiment.json').write_text(json.dumps(manifest,indent=2));(root/'pilot_processes.json').write_text(json.dumps(processes,indent=2))
print(json.dumps({'time':manifest['time'],'processes':processes},indent=2))
