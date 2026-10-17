"""Replay completed development traces; no simulator and no new evaluations."""
import fcntl,json,os,sys,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
b=Path('/home/user/xjy/nesy_vla_20261010');out=b/'program_causality_v79';out.mkdir(exist_ok=True)
sys.path.insert(0,str(b/'perceptual_fsm_v74/src'))
from nesy_vla.runtime import read,atomic,sha,setup
from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
setup(read(b/'symbolic_execution_v38/configs/grounded_outcome_feedback.json'))
from stripsnav.io import load_committed
torch.set_num_threads(2)
lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (out/'result.json').exists(),'Completed audit cannot be repeated'
state={'pid':os.getpid(),'state':'initializing','time':time.time()}
def progress(**fields):
 state.update(fields,time=time.time());atomic(out/'status.json',state);print(json.dumps(state),flush=True)
result={'started':time.time(),'scope':'Saved original-development observations only. Counterfactual action effects are conditional on recorded history, not new navigation success. No official fixed80 used.','arms':{}}
try:
 with torch.no_grad():
  for arm in ['observed_inverse']:
   run=b/'observation_inverse_v76'/arm;loc=run/'step_000040';saved=load_committed(loc)
   model=ProbabilisticTokenAutomaton(**saved['manifest']['architecture']).cuda().eval();model.load_state_dict(saved['model'])
   directory=run/'evaluations/step_000040';traces=[read(p) for p in sorted(directory.glob('*/trace.json'))]
   assert len(traces)==80 and read(directory/'summary.json')['episodes']==80
   counters={name:dict(actions=0,changed_actions=0,total_variation_sum=0.,changed_observed_states=0,changed_memory_states=0) for name in ['zero_program_coefficients','initial_program_reused','memory_reset','other_observation']}
   count=0;max_error=0.;repeated=0;repeat_pairs=0;collision_repeated=0;collision_pairs=0;coefficients=[]
   for trace in traces:
    for a,z in zip(trace['steps'],trace['steps'][1:]):
     repeated+=int(a['action']==z['action']);repeat_pairs+=1
     if a['collision']:collision_pairs+=1;collision_repeated+=int(a['action']==z['action'])
    for a,z in zip(trace['programs'],trace['programs'][1:]):
     coefficients.append(float(torch.linalg.vector_norm(torch.tensor(a['coefficients'])-torch.tensor(z['coefficients']))))
   requests=[]
   for step in range(32):
    selected=[t for t in traces if len(t['steps'])>step]
    requests.extend((step,selected[i:i+8]) for i in range(0,len(selected),8))
   for step,selected in requests:
    n=len(selected);selected=selected+[selected[-1]]*(8-n)
    rows=[t['steps'][step] for t in selected];start=(step//10)*10
    tasks=torch.tensor([['chair','table'].index(t['episode']['category']) for t in selected],device='cuda')
    cs=torch.tensor([next(p['coefficients'] for p in t['programs'] if p['start']==start) for t in selected],device='cuda')
    initial=torch.tensor([t['programs'][0]['coefficients'] for t in selected],device='cuda')
    x=torch.tensor([r['patch_tokens'] for r in rows],device='cuda')
    belief=torch.tensor([r['previous_belief'] for r in rows],device='cuda')
    action=torch.tensor([r['previous_action'] for r in rows],device='cuda')
    program=model.program_from_coefficients(cs,start,tasks);actual=model.step(program,x,belief,action,step)
    expected=torch.tensor([r['probabilities'] for r in rows],device='cuda')
    error=float((actual['probabilities']-expected).abs().max());max_error=max(max_error,error)
    assert error<2e-6,{'max_error':error,'step':step,'observed_id_mismatches':int((actual['observed_ids']!=torch.tensor([r['observation_tokens'] for r in rows],device='cuda')).sum()),'canonical_batch':8}
    assert actual['probabilities'].argmax(-1).tolist()==[r['action'] for r in rows]
    assert actual['observed_ids'].tolist()==[r['observation_tokens'] for r in rows]
    proof=model.proof(program,actual['observed_ids'],belief,action,actual['state_values'],step)
    assert torch.equal(proof['probabilities'],actual['probabilities'])
    _,reset=model.initial_state(8,'cuda')
    alternatives={
     'zero_program_coefficients':model.step(model.program_from_coefficients(torch.zeros_like(cs),start,tasks),x,belief,action,step),
     'initial_program_reused':model.step(model.program_from_coefficients(initial,start,tasks),x,belief,action,step),
     'memory_reset':model.step(program,x,reset,torch.full_like(action,17),step),
     'other_observation':model.step(program,x.roll(1,0),belief,action,step)}
    for name,r in alternatives.items():
     c=counters[name];c['actions']+=n;c['changed_actions']+=int((r['probabilities'].argmax(-1)!=actual['probabilities'].argmax(-1))[:n].sum())
     c['total_variation_sum']+=float((r['probabilities']-actual['probabilities'])[:n].abs().sum()/2)
     c['changed_observed_states']+=int((r['observed_ids']!=actual['observed_ids'])[:n].any(-1).sum())
     c['changed_memory_states']+=int((r['state_ids']!=actual['state_ids'])[:n].any(-1).sum())
    count+=n
   for c in counters.values():c['changed_action_fraction']=c['changed_actions']/c['actions'];c['mean_probability_tv']=c['total_variation_sum']/c['actions']
   result['arms'][arm]={'checkpoint_commit':read(loc/'COMMITTED'),'development_summary':read(directory/'summary.json'),
    'actions':count,'max_probability_replay_error':max_error,'repeat_actions':repeated,'repeat_pairs':repeat_pairs,
    'repeat_after_collision':collision_repeated,'collision_pairs':collision_pairs,'program_revision_l2_mean':sum(coefficients)/max(1,len(coefficients)),
    'counterfactuals':counters}
   progress(state='auditing',arm=arm,actions=count,max_probability_replay_error=max_error)
   del model,saved;torch.cuda.empty_cache()
 result['finished']=time.time();atomic(out/'result.json',result);progress(state='complete')
except BaseException as error:progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise
