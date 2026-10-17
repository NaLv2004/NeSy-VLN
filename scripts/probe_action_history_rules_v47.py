"""Matched rule-induction intervention on the previous-action shortcut.

No action replacement or manual predicate is introduced. Learned categorical
clauses remain the only action producer. Original fixed80 is never loaded into
fitting or evaluation. Completed episodes are never re-evaluated on resume.
"""
import argparse,fcntl,gc,hashlib,json,os,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np,torch
from PIL import Image
from scipy.sparse import csr_matrix
from sklearn.ensemble import ExtraTreesClassifier
base=Path('/home/user/xjy/nesy_vla_20261010');root=Path(__file__).resolve().parent
sys.path.insert(0,str(root));sys.path.insert(1,str(base/'symbolic_paired_v12'))
from nesy_vla.runtime import read,setup,DataBank,load_checkpoint,image_tensor,atomic,sha,pil
from nesy_vla.model import NavigationModel
from nesy_vla.evaluation import aggregate
from nesy_vla.token_rule_revision import TokenFacts,PackedRuleProgram,convert_v43_trees
p=argparse.ArgumentParser();p.add_argument('--arm',choices=['without_previous_action','action_dropout'],required=True);args=p.parse_args()
out=base/'action_history_ablation_v47'/args.arm;out.mkdir(parents=True,exist_ok=True)
lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (out/'status.json').exists() and read(out/'status.json')['state']=='complete':raise SystemExit('Already complete; no repeated evaluation')
torch.set_num_threads(2);torch.manual_seed(4300);schema=(64,)*49+(65,)*50
source=base/'spatial_automaton_v43';source_commit=read(source/'COMMITTED')
assert read(source/'status.json')['state']=='complete'
assert sha(source/'programs.json')==source_commit['program_sha256'] and sha(source/'visual_vocabulary.pt')==source_commit['vocabulary_sha256']
cfg=read(base/'symbolic_execution_v38/configs/grounded_outcome_feedback.json');setup(cfg);bank=DataBank(cfg)
assert bank.provenance==source_commit['data_provenance']
state={'pid':os.getpid(),'state':'initializing','time':time.time()}
def progress(**fields):state.update(fields,time=time.time());atomic(out/'status.json',state);print(json.dumps(state),flush=True)
manifest={'arm':args.arm,'source_commit':source_commit,'horizon':32,'sampling':'greedy','development_ids':[e['episode_id'] for e in bank.dev],'fixed80_used':False,'intervention':('Previous action is always the unknown token, including induction and deployment.' if args.arm=='without_previous_action' else 'Exactly30% of supervised training contexts have previous action replaced by unknown. Deployment receives the true previous action. Images, labels, rules, other hyperparameters unchanged.'),'memory':'49 current and49 previous learned visual tokens, one prior-action slot. No semantic labels or manually defined predicates.','rule_config':{'n_estimators':96,'max_depth':24,'min_samples_leaf':2,'max_features':.3,'random_state':4300},'limitations':'Diagnostic fixed induced programs; not an online ten-action program reviser. A successful alternative would still need integration into the learned reviser.'}
if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
else:atomic(out/'manifest.json',manifest)
sims=[]
try:
 if not (out/'COMMITTED').exists():
  records=read(source/'training_rows.json');codes=np.load(source/'visual_tokens.npy');labels=np.array([r['label'] for r in records]);supervised=np.array([r['supervised'] for r in records]);context=np.full((len(records),99),64,dtype=np.int16);context[:,:49]=codes
  for i in range(1,len(records)):
   if records[i]['episode_id']==records[i-1]['episode_id']:context[i,49:98]=codes[i-1];context[i,98]=labels[i-1]
  indices=np.flatnonzero(supervised)
  if args.arm=='without_previous_action':context[:,98]=64;changed=indices
  else:
   changed=np.random.default_rng(4700).choice(indices,int(.3*len(indices)),replace=False);context[changed,98]=64
  atomic(out/'input_intervention.json',{'training_rows_sha256':sha(source/'training_rows.json'),'token_file_sha256':sha(source/'visual_tokens.npy'),'changed_training_indices':changed.tolist(),'development_training_overlap':sorted(set(r['episode_id'] for r in records)&set(manifest['development_ids']))})
  assert not set(r['episode_id'] for r in records)&set(manifest['development_ids'])
  columns=(np.arange(99)[None]*65+context).ravel();X=csr_matrix((np.ones(len(columns)),(np.repeat(np.arange(len(context)),99),columns)),shape=(len(context),99*65));programs={};training={}
  for task in sorted({r['category'] for r in records}):
   selected=np.array([i for i,r in enumerate(records) if r['category']==task and r['supervised']]);progress(state='inducing_rules',task=task,rows=len(selected))
   learner=ExtraTreesClassifier(**manifest['rule_config'],n_jobs=2);learner.fit(X[selected],labels[selected]);trees=[]
   for est in learner.estimators_:
    t=est.tree_;values=t.value[:,0].copy();values/=values.sum(1,keepdims=True);conclusions=np.zeros((t.node_count,17));conclusions[:,learner.classes_.astype(int)]=values
    assert ((t.threshold[t.feature>=0]>=0)&(t.threshold[t.feature>=0]<1)).all()
    if args.arm=='without_previous_action':assert not (t.feature[t.feature>=0]//65==98).any()
    trees.append({'predicate':t.feature.tolist(),'false_branch':t.children_left.tolist(),'true_branch':t.children_right.tolist(),'conclusion':[c.tolist() if t.feature[j]<0 else None for j,c in enumerate(conclusions)]})
   programs[task]=trees;packed=PackedRuleProgram(convert_v43_trees(trees,schema),schema)
   expected=np.zeros((min(200,len(selected)),17));expected[:,learner.classes_.astype(int)]=learner.predict_proba(X[selected[:200]])
   evidence,_=packed.evidence(TokenFacts(torch.from_numpy(context[selected[:200]].astype(np.int64)),schema));error=float(np.max(np.abs(evidence.mean(1)-expected)));assert error<1e-10
   training[task]={'rows':len(selected),'expert_accuracy':float((learner.predict(X[selected])==labels[selected]).mean()),'serialization_probability_error':error,'nodes':sum(len(t['predicate']) for t in trees)}
   del packed,learner;gc.collect()
  atomic(out/'programs.json',programs,compact=True);atomic(out/'training_metrics.json',training);atomic(out/'COMMITTED',{'program_sha256':sha(out/'programs.json'),'vocabulary_sha256':source_commit['vocabulary_sha256'],'training_input_sha256':sha(out/'input_intervention.json'),'manifest_sha256':sha(out/'manifest.json'),'time':time.time()});del context,X,records,codes;gc.collect()
 else:
  assert sha(out/'programs.json')==read(out/'COMMITTED')['program_sha256'];programs=read(out/'programs.json')
 programs={task:PackedRuleProgram(convert_v43_trees(t,schema),schema) for task,t in programs.items()}
 vision_source=base/'symbolic_conjunctive_v23/runs/conjunctive/best_development';vision_cfg=read(vision_source.parent/'config.json');model=NavigationModel(vision_cfg).eval();load_checkpoint(vision_source,model);model.requires_grad_(False).to('cuda:0');vision=model.grounder.vision;del model;gc.collect()
 vocabulary=torch.load(source/'visual_vocabulary.pt',map_location='cuda:0',weights_only=True)
 @torch.no_grad()
 def encode(image):
  tensor=image_tensor(image,vision_cfg['image_size'],'cuda:0')
  with torch.autocast('cuda',dtype=torch.bfloat16):f=torch.nn.functional.adaptive_avg_pool2d(vision(tensor),(7,7)).flatten(2).transpose(1,2)
  v=torch.nn.functional.normalize(((f.float()-vocabulary['mean'])/vocabulary['sd'])@vocabulary['axes'],dim=-1)
  return torch.cdist(v.flatten(0,1),vocabulary['centers']).argmin(1).cpu().long()
 # Reuse existing frames to verify the exact evaluated source encoding path.
 for marker in sorted((source/'development').glob('*/evaluation.json'))[:3]:
  t=read(marker.parent/'trace.json');assert encode(Image.open(marker.parent/'first.png').convert('RGB')).tolist()==t['steps'][0]['symbols'][:49]
 sims=[bank.simulator(out/f'simulator_{i}.log',0,evaluation=True) for i in range(4)];atomic(out/'processes.json',{'pid':os.getpid(),'simulator_pids':[s.process.pid for s in sims],'time':time.time()})
 done=[];pending=[]
 for e in bank.dev:
  marker=out/'development'/e['episode_id']/'evaluation.json'
  if marker.exists():done.append(read(marker))
  else:pending.append(e)
 with ThreadPoolExecutor(max_workers=4) as pool:
  for offset in range(0,len(pending),4):
   episodes=pending[offset:offset+4];responses=list(pool.map(lambda pair:pair[0].call('reset',episode=pair[1]),zip(sims,episodes)));images=[pil(r['rgb']) for r in responses];trials=[]
   for e,img in zip(episodes,images):
    folder=out/'development'/e['episode_id'];folder.mkdir(parents=True,exist_ok=True);img.save(folder/'first.png');trials.append({'episode':e,'steps':[],'previous':torch.full((49,),64,dtype=torch.long),'last_action':64})
   active=list(range(len(episodes)))
   for step in range(32):
    actions=[]
    for i in active:
     r=trials[i];code=encode(images[i]);previous_action=64 if args.arm=='without_previous_action' else r['last_action'];facts=torch.cat([code,r['previous'],torch.tensor([previous_action])]);evidence,leaves=programs[r['episode']['category']].evidence(TokenFacts(facts[None],schema));p=evidence[0].mean(0);action=int(p.argmax())
     r['steps'].append({'step':step,'symbols':facts.tolist(),'action':action,'probabilities':p.tolist(),'leaves':leaves[0].tolist(),'image_sha256':hashlib.sha256(np.asarray(images[i]).tobytes()).hexdigest()});r['previous']=code;r['last_action']=action;actions.append(action)
    responses=list(pool.map(lambda pair:sims[pair[0]].call('step',action=pair[1]),zip(active,actions)));remaining=[]
    for i,response in zip(active,responses):
     trials[i]['steps'][-1]['collision']=bool(response.get('feedback',{}).get('collision',False));trials[i]['metrics']=response.get('metrics',{});images[i]=pil(response['rgb'])
     if not response['done']:remaining.append(i)
    active=remaining
    if not active:break
   for r,image in zip(trials,images):
    e=r['episode'];folder=out/'development'/e['episode_id'];image.save(folder/'last.png');atomic(folder/'trace.json',{'episode':e,'metrics':r['metrics'],'steps':r['steps']},compact=True)
    row={'episode_id':e['episode_id'],'scene':e['scene'],'domain':e['domain'],'category':e['category'],'distance_group':e['distance_group'],'success':bool(r['metrics'].get('success',False)),'steps':len(r['steps']),'collisions':sum(s['collision'] for s in r['steps'])};atomic(folder/'evaluation.json',row);done.append(row)
   progress(state='evaluating_development',summary=aggregate(done))
 atomic(out/'development/summary.json',aggregate(done));progress(state='complete',summary=aggregate(done))
except BaseException as error:progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise
finally:
 for sim in sims:sim.close()
