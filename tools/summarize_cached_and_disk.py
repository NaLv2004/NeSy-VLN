import json,subprocess
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010')
for version in ['symbolic_robust_v7','symbolic_loss_v6']:
    d=json.loads((base/version/'current_status.json').read_text());print(version,'time',d['time'])
    for name,j in d['jobs'].items():
        ranks=j.get('ranks',[])
        print(json.dumps({'name':name,'status':j['status'],'live':bool(j['live_command']),'checkpoint':j['checkpoint'],
            'metrics':[{k:r['losses'].get(k) for k in ['expert_action_accuracy','changed_symbol_bit_accuracy','inverse_accuracy','view_bit_disagreement_metric']} for r in ranks],
            'eval':{step:{suite:({k:v.get(k) for k in ['successes','episodes','collision_action_rate','mean_steps']} if v else None) for suite,v in suites.items()} for step,suites in j['evaluations'].items()}}))
print(subprocess.check_output(['du','-x','-B1','--max-depth=3',str(base)],text=True)[-16000:])
print(subprocess.check_output(['df','-h',str(base)],text=True))
