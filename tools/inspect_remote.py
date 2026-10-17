import json, os, subprocess
from pathlib import Path

root = Path('/home/user/xjy/strips_wm_navigation_20261009')
obsolete=[]
for marker in root.joinpath('runs').rglob('COMMITTED'):
    try:
        data=json.loads(marker.read_text())
        if 'checkpoint_file' not in data: continue
        current=(marker.parent/data['checkpoint_file']).resolve()
        if not current.is_file(): continue
        for file in marker.parent.glob('checkpoint.*.pt'):
            if file.is_symlink() or file.resolve()==current: continue
            obsolete.append({'path':str(file),'bytes':file.stat().st_size,'current':str(current),
                'current_sha256':data.get('checkpoint_sha256')})
    except (ValueError,OSError): pass
print(json.dumps({'obsolete_generation_count':len(obsolete),'bytes':sum(x['bytes'] for x in obsolete),
                  'sample':obsolete[:8]},indent=2))
root.joinpath('research/obsolete_generations_20261010.json').write_text(json.dumps(obsolete,indent=2))
for base in [Path('/home/user/xjy/models'),Path('/home/user/xjy/lite3_navigation'),Path('/home/user/xjy/qwen35_vlm_20260920')]:
    if base.exists(): print('MODEL_ROOT',str(base),[str(p) for p in base.glob('*')][:20])
for config in [Path('/home/user/xjy/lite3_photo_world_20260927/lab_rl64/run32_v4_photo25/config.json')]:
    c=json.loads(config.read_text()); print('MODEL',c.get('model'))
rows=json.loads(root.joinpath('data/search_cost_supervision_v2/examples.json').read_text())
print('EXAMPLE',len(rows),rows[:2])
for file in root.joinpath('data/transitions').glob('*'):
    print('TRANSITIONS',file.name); break
print(subprocess.getoutput('/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python -c "import torch,transformers,peft; print(torch.__version__,transformers.__version__,peft.__version__)"'))
