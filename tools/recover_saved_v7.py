import json,time,subprocess
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010')
def read(p):return json.loads(p.read_text()) if p.exists() else None
def live(pid):
    p=Path(f'/proc/{pid}/cmdline')
    return p.read_bytes().replace(b'\0',b' ').decode() if p.exists() else None
for rel in ['data/lossless_demonstrations_v7/inspection.json','symbolic_loss_v6/current_status.json','symbolic_sft_v5/current_status.json']:
    obj=read(root/rel)
    if obj and 'jobs' in obj and isinstance(obj['jobs'],dict):
        obj={k:v for k,v in obj.items() if k in ['time','stable','gpu','jobs']}
        for name,j in obj['jobs'].items():
            obj['jobs'][name]={k:v for k,v in j.items() if k in ['status','training','checkpoint','evaluations','live_command']}
            for ev in obj['jobs'][name].get('evaluations',{}).values():
                for suite,val in ev.items():
                    if val:ev[suite]={k:v for k,v in val.items() if k in ['success_rate','successes','episodes','collision_rate','mean_steps']}
    print(json.dumps({'file':rel,'saved':obj}))
for rel in ['data/lossless_demonstrations_v7/controller.json','data/lossless_demonstrations_v7/READY.json','diagnostics_20261011/demonstration_replay/result.json']:
    obj=read(root/rel)
    if rel.endswith('controller.json') and obj:
        obj['controller_live']=live(obj['pid'])
        for job in obj.get('jobs',[]):job['actual_live']=live(job['pid'])
    if 'demonstration_replay' in rel and obj:obj={k:v for k,v in obj.items() if k!='episodes'}
    print(json.dumps({'file':rel,'data':obj}))
print(json.dumps({'server_time':time.time()}))
