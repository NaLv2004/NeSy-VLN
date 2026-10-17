"""Audit existing traces only: rule fidelity, paired outcomes and failure loops."""
import gc,json,sys,time
from pathlib import Path
import numpy as np,torch
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'recovery_rule_learning_v49'
sys.path.insert(0,str(out/'src'))
from nesy_vla.runtime import read,sha,atomic
from nesy_vla.token_rule_revision import TokenFacts,PackedRuleProgram,convert_v43_trees
schema=(64,)*49+(65,)*50;torch.set_num_threads(2)
roots={'original':b/'spatial_automaton_v43','no_action':b/'action_history_ablation_v47/without_previous_action',
       'corrected_no_action':out/'without_previous_action'}
results={};outcomes={}
for name,root in roots.items():
 assert read(root/'status.json')['state']=='complete'
 traces=[read(p) for p in sorted((root/'development').glob('*/trace.json'))];assert len(traces)==80
 outcomes[name]={t['episode']['episode_id']:bool(t['metrics'].get('success',False)) for t in traces}
 summary={'episodes':80,'successes':sum(outcomes[name].values()),'actions':0,'collisions':0,'adjacent_pairs':0,
  'repeated_actions':0,'after_collision_pairs':0,'repeated_after_collision':0,'same_current_tokens':0,'failed_stops':0}
 for t in traces:
  rows=t['steps'];summary['actions']+=len(rows);summary['collisions']+=sum(r['collision'] for r in rows)
  summary['failed_stops']+=int(not t['metrics'].get('success',False) and rows[-1]['action']==0)
  for previous,current in zip(rows,rows[1:]):
   summary['adjacent_pairs']+=1;summary['repeated_actions']+=int(previous['action']==current['action'])
   summary['same_current_tokens']+=int(previous['symbols'][:49]==current['symbols'][:49])
   if previous['collision']:
    summary['after_collision_pairs']+=1;summary['repeated_after_collision']+=int(previous['action']==current['action'])
 summary['collision_rate']=summary['collisions']/summary['actions']
 results[name]=summary
root=roots['corrected_no_action'];commit=read(root/'COMMITTED');assert sha(root/'programs.json')==commit['program_sha256']
raw=read(root/'programs.json');programs={k:PackedRuleProgram(convert_v43_trees(v,schema),schema) for k,v in raw.items()};del raw;gc.collect()
error=0.;count=0
for path in sorted((root/'development').glob('*/trace.json')):
 t=read(path);codes=torch.tensor([r['symbols'] for r in t['steps']]);assert (codes[:,-1]==64).all()
 evidence,leaves=programs[t['episode']['category']].evidence(TokenFacts(codes,schema));p=evidence.mean(1)
 expected=np.array([r['probabilities'] for r in t['steps']]);error=max(error,float(np.max(np.abs(p-expected))))
 assert np.array_equal(leaves,np.array([r['leaves'] for r in t['steps']]))
 assert np.array_equal(p.argmax(-1),np.array([r['action'] for r in t['steps']]))
 count+=len(p)
assert error<1e-10
paired={}
for name in ['original','no_action']:
 old=outcomes[name];new=outcomes['corrected_no_action'];assert set(old)==set(new)
 paired[name]={'gained':[k for k in new if new[k] and not old[k]],'lost':[k for k in new if old[k] and not new[k]],
               'both_success':sum(old[k] and new[k] for k in new),'both_failure':sum(not old[k] and not new[k] for k in new)}
result={'time':time.time(),'summaries':results,'paired_outcomes':paired,'proof':{'actions':count,'maximum_probability_error':error,
 'all_selected_actions_reproduced_from_integer_symbols_and_induced_rules':True,'checkpoint':commit},
 'fixed80_used':False,'fresh_simulator_rollouts':0,
 'qualification':'Original80 development comparisons. Corrected data are training-only. Static-rule gains do not complete the requested online learned state-machine framework.'}
atomic(out/'behavior_and_proof_audit.json',result)
print(json.dumps({**result,'paired_outcomes':{k:{'gained_count':len(v['gained']),'lost_count':len(v['lost']),'both_success':v['both_success'],'both_failure':v['both_failure']} for k,v in paired.items()}},indent=2))
