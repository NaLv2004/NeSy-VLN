"""Prepare matched PPO from the development-selected, discrete-only v76 policy."""
from pathlib import Path

root=Path(__file__).resolve().parents[1]
s=(root/'scripts/train_full_automaton_rl_v62.py').read_text(encoding='utf-8')
s=s.replace('full_automaton_rl_v62','perceptual_rl_v80')
s=s.replace("p.add_argument('--updates',type=int,default=80);args=p.parse_args()", "p.add_argument('--updates',type=int,default=40);p.add_argument('--verify-only',action='store_true');args=p.parse_args()")
start=s.index('from stripsnav.io import committed_torch_save,load_committed')
end=s.index('\ngroups={}',start)
s=s[:start]+'''from stripsnav.io import load_committed
from nesy_vla.small_checkpoint import save_small_checkpoint
from nesy_vla.observation_inverse import observation_inverse_logits
def committed_torch_save(folder,payload,metadata):
    return save_small_checkpoint(folder,payload,metadata,reservation_root=base/'perceptual_rl_v80')
source=base/'perceptual_sequences_v73/tail_1e4';extra=base/'data/onpolicy_recovery_v48'
source_commit=read(source/'TOKENS_COMMITTED');extra_commit=read(extra/'COMPLETE')
assert bank.provenance==source_commit['data_provenance']
assert sha(source/'visual_tokens.npy')==source_commit['tokens_sha256']
assert sha(source/'training_rows.json')==source_commit['training_rows_sha256']
assert sha(source_commit['vocabulary_path'])==source_commit['vocabulary_sha256']
rows=read(source/'training_rows.json');old_count=source_commit['old_count']
codes=np.load(source/'visual_tokens.npy')
assert codes.shape==(len(rows),49) and codes.min()>=0 and codes.max()<64
'''+s[end:]
s=s.replace("base/'probabilistic_automaton_v56/soft_recurrent/step_000040'", "base/'observation_inverse_v76/observed_inverse/step_000040'")
s=s.replace("warm=load_committed(warm_source);architecture=warm['manifest']['architecture']", "assert read(warm_source/'COMMITTED')['checkpoint_sha256']=='a842c20cb147822b6b5321fb79247771009a5d8a3b599e5d2979e2cbc940723e'\nwarm=load_committed(warm_source);architecture=warm['manifest']['architecture']\nassert warm['manifest']['source_commit']==source_commit")
s=s.replace('AdamW(trainable,lr=1e-4','AdamW(trainable,lr=3e-5').replace("'actor_lr':1e-4", "'actor_lr':3e-5")
s=s.replace("'optimizer_resume':False,'source_sft_optimizer_retained':True}", "'optimizer_resume':False,'source_sft_optimizer_retained':True,'source_commit':source_commit,\n 'selection':'Immutable v76 observed_inverse40 selected on complete development only; no source baseline navigation rerun.',\n 'inverse_auxiliary':'Observed hard IDs under the same program; no recurrent action-bearing belief enters inverse prediction.',\n 'storage':'Small full-resume atomic checkpoints with 512MiB plus twice serialized size reserve.'}")
s=s.replace("        if before_state is not None:\n            inverse=model.inverse_action(before_state,r['state_values'])", "        if step:\n            inverse=observation_inverse_logits(model,program,x[:,step-1],x[:,step],step)")
start=s.index('def initialize_evaluation():')
end=s.index('\n@torch.no_grad()\ndef encode',start)
s=s[:start]+'''def initialize_evaluation(launch_simulators=True):
    global vision,vision_cfg,vocabulary,sims
    if vision is not None:return
    from nesy_vla.model import NavigationModel
    location=Path(source_commit['encoder_source']);vision_cfg=read(location.parent/'config.json')
    assert read(location/'COMMITTED')==source_commit['encoder_source_commit']
    visual=NavigationModel(vision_cfg).eval();load_checkpoint(location,visual);visual.requires_grad_(False).cuda()
    vision=visual.grounder.vision;del visual
    if hasattr(vision,'backbone'):vision=vision.backbone
    assert isinstance(vision,torch.nn.Sequential) and len(vision)==8
    component=Path(source_commit['component_checkpoint']);assert read(component/'COMMITTED')==source_commit['component_commit']
    payload=load_committed(component);vision[6:].load_state_dict(payload['visual_tail']);del payload;gc.collect()
    vocabulary=torch.load(source_commit['vocabulary_path'],map_location='cuda',weights_only=True)
    for i in [0,len(rows)//2,len(rows)-1]:
        with Image.open(rows[i]['path']) as image:assert encode(image.convert('RGB')).tolist()==codes[i].tolist()
    if launch_simulators:
        sims=[bank.simulator(out/f'sim_{j}.log',0,evaluation=True) for j in range(8)]
        atomic(out/'processes.json',{'pid':os.getpid(),'simulator_pids':[s.process.pid for s in sims],'time':time.time()})
'''+s[end:]
needle='try:\n initialize_evaluation()'
verify='''try:
 initialize_evaluation(launch_simulators=not args.verify_only)
 if args.verify_only:
  saved=warm_source.parent/'evaluations/step_000040'
  traces=[read(saved/e['episode_id']/'trace.json') for e in bank.dev[:8]]
  with torch.no_grad():
   grids=[]
   for e,t in zip(bank.dev[:8],traces):
    with Image.open(saved/e['episode_id']/'first.png') as image:grids.append(encode(image.convert('RGB')))
    assert grids[-1].tolist()==t['steps'][0]['patch_tokens']
   x=torch.stack(grids);task=torch.tensor([tasks.index(e['category']) for e in bank.dev[:8]],device='cuda')
   _,memory=model.initial_state(8,'cuda');previous=torch.full((8,),17,device='cuda',dtype=torch.long)
   program=model.generate(task,x[:,None],0);result=model.step(program,x,memory,previous,0)
   expected=torch.tensor([t['steps'][0]['probabilities'] for t in traces],device='cuda')
   probability_error=float((result['probabilities']-expected).abs().max());assert probability_error==0
   proof=model.proof(program,result['observed_ids'],memory,previous,result['state_values'],0)
   assert torch.equal(proof['probabilities'],result['probabilities'])
  # Teacher gradient integration uses original training demonstrations only.
  loss=demonstration_loss(sorted(episodes,key=lambda e:len(e['indices']),reverse=True)[:8])
  optimizer.zero_grad(set_to_none=True);loss.backward()
  gradients={name:float(param.grad.norm()) for name,param in model.named_parameters() if param.grad is not None}
  assert gradients and all(np.isfinite(v) for v in gradients.values()) and sum(gradients.values())>0
  if args.arm=='action_rules_only':assert set(gradients)<=set(manifest['trainable_names'])
  verified={'time':time.time(),'arm':args.arm,'source_sha256':manifest['warm_start_commit']['checkpoint_sha256'],
    'training_frontend_checks':3,'saved_development_first_frames':8,'probability_error':probability_error,
    'full_training_sequence_loss':float(loss.detach()),'gradient_norms':gradients,'fresh_rollouts':0,
    'fixed80_used':False,'optimizer_updates':0,'complete':True}
  atomic(out/'verification.json',verified);print(json.dumps(verified,indent=2));raise SystemExit(0)'''
assert needle in s;s=s.replace(needle,verify)
# Successful verification must not be recorded as a failed run.
s=s.replace('except BaseException as error:', 'except Exception as error:')
(root/'scripts/train_perceptual_rl_v80.py').write_text(s,encoding='utf-8')
compile(s,'train_perceptual_rl_v80.py','exec')
print('Generated PPO experiment with matching perceptual frontend and action-independent inverse supervision.')
