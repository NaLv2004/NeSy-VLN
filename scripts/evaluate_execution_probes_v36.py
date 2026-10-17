"""Full isolated development evaluation of the two completed symbol-learning probes."""
import fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_paired_v12';sys.path.insert(0,str(root))
from nesy_vla.runtime import read,setup,DataBank,load_checkpoint,atomic,sha
from nesy_vla.model import NavigationModel
from nesy_vla.policy import rollout
from nesy_vla.reporting import save_trace
from nesy_vla.evaluation import aggregate
from nesy_vla.diagnostics import audit_policies
from nesy_vla.outcome_supervision import SymbolicExecutionDecoder

torch.set_num_threads(2);torch.manual_seed(3600)
out=base/'execution_development_v36';out.mkdir(exist_ok=True)
handle=(out/'run.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
source=base/'symbolic_conjunctive_v23/runs/conjunctive/best_development'
cfg=read(source.parent/'config.json');cfg['trace_image_store']=str(base/'evidence_archives/live_png_store');setup(cfg)
bank=DataBank(cfg);assert len(bank.dev)==80 and cfg['horizon']==32 and cfg['revision_interval']==10
assert read(base/'execution_learning_v34/result.json')['state']=='complete'
manifest={'source':str(source),'source_commit':read(source/'COMMITTED'),
    'development_episode_ids':[e['episode_id'] for e in bank.dev],
    'development_sha256':hashlib.sha256(json.dumps(bank.dev,sort_keys=True).encode()).hexdigest(),
    'arms':{name:{'delta':str(base/'execution_learning_v34'/f'{name}_diagnostic_weights.pt'),
        'delta_sha256':sha(base/'execution_learning_v34'/f'{name}_diagnostic_weights.pt')} for name in ['bc_control','execution_feedback']},
    'horizon':32,'revision_interval':10,'sampling':'greedy','fixed80_used':False,
    'policy_device':'cpu','simulator_gpu':0,'auxiliary_decoder_removed_before_runtime':True}
if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest,'Frozen evaluation identity changed'
else:atomic(out/'manifest.json',manifest)
status={'state':'starting','pid':os.getpid(),'time':time.time(),'arms':{}}
atomic(out/'status.json',status);sim=None
try:
    free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<5500:raise RuntimeError('Insufficient spare simulator memory; leave training untouched')
    sim=bank.simulator(out/'simulator.log',0,evaluation=True)
    atomic(out/'processes.json',{'pid':os.getpid(),'simulator_pid':sim.process.pid,'time':time.time()})
    for name,record in manifest['arms'].items():
        destination=out/name;destination.mkdir(exist_ok=True)
        model=NavigationModel(cfg).eval();load_checkpoint(source,model)
        model.execution_decoder=SymbolicExecutionDecoder(cfg['program']['d'],cfg['program']['a'])
        assert sha(record['delta'])==record['delta_sha256']
        loaded=model.load_state_dict(torch.load(record['delta'],map_location='cpu',weights_only=True),strict=False)
        assert not loaded.unexpected_keys and all(k.startswith(('grounder.vision.','teacher.')) for k in loaded.missing_keys)
        del model.execution_decoder
        model.requires_grad_(False);rows=[]
        for episode in bank.dev:
            folder=destination/episode['episode_id'];marker=folder/'evaluation.json'
            if marker.exists():row=read(marker)
            else:
                with torch.no_grad():policy,metrics=rollout(model,sim,episode,stochastic=False)
                proof=audit_policies(model,[policy]);assert proof['discrete_boundary'] and proof['replay_probability_max_error']<1e-5
                save_trace(folder,policy,episode,metrics)
                steps=[r for c in policy.chunks for r in c['rows']]
                row={'episode_id':episode['episode_id'],'scene':episode['scene'],'domain':episode['domain'],
                    'category':episode['category'],'distance_group':episode['distance_group'],
                    'success':bool(metrics.get('success')),'steps':len(steps),'collisions':sum(r['collision'] for r in steps),
                    'repeated_action_pairs':sum(a['action']==b['action'] for a,b in zip(steps,steps[1:])),
                    'revision_count':max(0,len(policy.chunks)-1),'symbolic_audit':proof,
                    'first_frame_sha256':sha(folder/'frame_000.png')}
                other=out/'bc_control'/episode['episode_id']/'evaluation.json'
                if name!='bc_control' and other.exists():assert row['first_frame_sha256']==read(other)['first_frame_sha256']
                atomic(marker,row)
            rows.append(row);status.update(state='evaluating',time=time.time(),current_arm=name)
            status['arms'][name]=aggregate(rows);atomic(out/'status.json',status)
        atomic(destination/'summary.json',aggregate(rows));del model
    status.update(state='complete',time=time.time());atomic(out/'status.json',status)
except BaseException as error:
    status.update(state='failed',time=time.time(),error=repr(error),traceback=traceback.format_exc());atomic(out/'status.json',status);raise
finally:
    if sim is not None:sim.close()
print(json.dumps(status,indent=2),flush=True)
