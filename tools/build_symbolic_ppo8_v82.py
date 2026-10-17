"""Expand the validated actor to one synchronous eight-GPU PPO experiment."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'scripts/train_perceptual_rl_v80.py').read_text(encoding='utf-8')
s=s.replace('import argparse,fcntl,gc,hashlib,json,os,random,sys,time,traceback', 'import argparse,datetime,fcntl,gc,hashlib,json,os,random,sys,time,traceback\nimport torch.distributed as dist')
s=s.replace("p.add_argument('--updates',type=int,default=40)","p.add_argument('--updates',type=int,default=120)")
s=s.replace("out=base/'perceptual_rl_v80'/args.arm;out.mkdir(parents=True,exist_ok=True)","assert args.arm=='joint','One primary experiment only'\nrank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])\ntorch.cuda.set_device(0);dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=30))\nout=base/'symbolic_ppo8_v82';rank_dir=out/f'rank{rank}';rank_dir.mkdir(parents=True,exist_ok=True)")
s=s.replace("lock=(out/'run.lock')", "lock=(rank_dir/'run.lock')")
s=s.replace("rng=random.Random(5100)","rng=random.Random(5100+1009*rank)")
s=s.replace("from nesy_vla.observation_inverse import observation_inverse_logits", "from nesy_vla.distributed_symbolic import global_moments,average_gradients,gather_objects,assert_identical_parameters\nfrom nesy_vla.sequence_perception import SequencePerception")
s=s.replace("reservation_root=base/'perceptual_rl_v80'", "reservation_root=base/'symbolic_ppo8_v82'")
s=s.replace("base/'observation_inverse_v76/observed_inverse/step_000040'", "base/'sequence_perception_v78/frozen/step_000005'")
start=s.index("assert read(warm_source/'COMMITTED')['checkpoint_sha256']==")
end=s.index("warm=load_committed(warm_source)",start)
s=s[:start]+"assert read(warm_source/'COMMITTED')==read(out/'SOURCE.json')['commit']\n"+s[end:]
s=s.replace("'canonical_symbolic_batch':8", "'canonical_symbolic_batch':8,'world_size':world,'global_episodes_per_update':16*world")
s=s.replace("evaluations/step_000040/summary.json", "evaluations/step_000005/summary.json")
s=s.replace("'comparison':'Identical source, task sampler and sampling seed. Joint trains all learned states/rules/generator; control trains only action-conclusion tables and rule strengths.'", "'comparison':'No control arms. All GPUs jointly optimize one actor, grounding, definitions, transitions and dynamic-program generator.'")
s=s.replace("'selection':'Immutable v76 observed_inverse40 selected on complete development only; no source baseline navigation rerun.'", "'selection':'Development-selected v78 frozen5, 37/80; no fixed80 selection or duplicate baseline rollouts.'")
s=s.replace("'inverse_auxiliary':'Observed hard IDs under the same program; no recurrent action-bearing belief enters inverse prediction.'", "'inverse_auxiliary':'Disabled as in the selected supervised source; no action-bearing belief inverse shortcut.'")
s=s.replace("else:atomic(out/'manifest.json',manifest)","elif rank==0:atomic(out/'manifest.json',manifest)\ndist.barrier()")
s=s.replace("state={'pid':os.getpid()", "state={'rank':rank,'pid':os.getpid()")
s=s.replace("atomic(out/'status.json',state)", "atomic(rank_dir/'status.json',state)")
start=s.index('begin=0')
end=s.index('all_codes=torch.from_numpy',start)
s=s[:start]+'''torch.manual_seed(6100+rank*1009)
begin=0
if (out/'latest/COMMITTED').exists():
 saved=load_committed(out/'latest');assert saved['manifest']==manifest
 model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
 critic.load_state_dict(saved['critic']);value_optimizer.load_state_dict(saved['value_optimizer']);begin=saved['epoch']
 state_rng=saved['rank_rng'][rank];rng.setstate(state_rng['python']);torch.set_rng_state(state_rng['torch']);torch.cuda.set_rng_state(state_rng['cuda'])
 del saved
assert_identical_parameters(model);assert_identical_parameters(critic)

def save(folder,epoch):
 rngs=gather_objects({'python':rng.getstate(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()})
 if rank==0:
  payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'critic':critic.state_dict(),
    'value_optimizer':value_optimizer.state_dict(),'epoch':epoch,'manifest':manifest,'rank_rng':rngs,
    'visual_last_block':perception.block.state_dict()}
  committed_torch_save(folder,payload,{'epoch':epoch,'horizon':32,'revision_interval':10,'optimizer_included':True,'ranks':world})
 dist.barrier()
'''+s[end:]
s=s.replace("atomic(out/'training_data.json'", "atomic(rank_dir/'training_data.json'")
start=s.index('        if step:\n            inverse=')
end=s.index('        for distribution',start)
s=s[:start]+s[end:]
s=s.replace('vision=None;sims=[];vision_cfg=None;vocabulary=None','vision=None;sims=[];vision_cfg=None;vocabulary=None;prefix=None;perception=None')
s=s.replace('global vision,vision_cfg,vocabulary,sims','global vision,vision_cfg,vocabulary,sims,prefix,perception')
start=s.index("    for i in [0,len(rows)//2,len(rows)-1]:", s.index('def initialize_evaluation'))
end=s.index('    if launch_simulators:',start)
s=s[:start]+'''    prefix=torch.nn.Sequential(*list(vision.children())[:7],torch.nn.Sequential(*list(vision[7].children())[:-1])).eval().requires_grad_(False)
    perception=SequencePerception(vision[7][-1],vocabulary,canonical_batch=16).cuda().eval().requires_grad_(False)
    saved=load_committed(warm_source);perception.block.load_state_dict(saved['visual_last_block']);del saved
    # Re-encode training data using exactly the selected frontend. Only integer
    # codes are exchanged across ranks; no visual features reach the actor.
    encoded_path=out/'training_codes.npy'
    if not args.verify_only:
        if encoded_path.exists():
            meta=read(out/'TRAINING_CODES_COMMITTED');assert sha(encoded_path)==meta['sha256'] and meta['source_commit']==read(warm_source/'COMMITTED')
            all_codes.copy_(torch.from_numpy(np.load(encoded_path).astype(np.int64)).cuda())
        else:
            all_codes.zero_()
            for i in range(rank,len(rows),world):
                with Image.open(rows[i]['path']) as image:all_codes[i].copy_(encode(image.convert('RGB')))
            dist.all_reduce(all_codes)
            if rank==0:
                temporary=out/'training_codes.partial.npy';np.save(temporary,all_codes.cpu().numpy().astype(np.uint8));os.replace(temporary,encoded_path)
                atomic(out/'TRAINING_CODES_COMMITTED',{'source_commit':read(warm_source/'COMMITTED'),'sha256':sha(encoded_path),'rows':len(rows),'training_only':True})
            dist.barrier()
        for i in [0,len(rows)//2,len(rows)-1]:
            with Image.open(rows[i]['path']) as image:assert torch.equal(encode(image.convert('RGB')),all_codes[i])
'''+s[end:]
s=s.replace("out/f'sim_{j}.log'", "rank_dir/f'sim_{j}.log'").replace("atomic(out/'processes.json'", "atomic(rank_dir/'processes.json'")
start=s.index('    with torch.autocast',s.index('def encode(image):'))
end=s.index('\n@torch.no_grad()\ndef evaluate',start)
s=s[:start]+'''    with torch.autocast('cuda',dtype=torch.bfloat16):features=prefix(image_tensor(image,vision_cfg['image_size'],'cuda'))
    return perception(features).ids[0]
'''+s[end:]
s=s.replace('    for e in bank.dev:\n', '    for i,e in enumerate(bank.dev):\n        if (i//8)%world!=rank:continue\n')
start=s.index("    summary=aggregate(done);atomic(directory/'summary.json',summary)")
end=s.index('\n@torch.no_grad()\ndef collect',start)
s=s[:start]+'''    done=[row for part in gather_objects(done) for row in part]
    assert len(done)==80 and len({r['episode_id'] for r in done})==80
    summary=aggregate(done)
    flags=torch.tensor([proof_error],device='cuda');dist.all_reduce(flags,op=dist.ReduceOp.MAX)
    assert float(flags[0])==0
    if rank==0:
        atomic(directory/'summary.json',summary)
        atomic(directory/'proof_audit.json',{'maximum_action_probability_error':float(flags[0]),'all_actions_have_discrete_transition_and_rule_proofs':True,'episodes':80})
    dist.barrier()
    best=read(out/'best_score.json') if (out/'best_score.json').exists() else {'success_rate':-1}
    if summary['success_rate']>best['success_rate']:
        save(out/'best_development',epoch)
        if rank==0:atomic(out/'best_score.json',dict(summary,epoch=epoch))
    dist.barrier();model.train();return summary
'''+s[end:]
s=s.replace("saved=warm_source.parent/'evaluations/step_000040'", "saved=warm_source.parent/'evaluations/step_000005'")
# Preflight uses a real, disjoint training sequence per worker and checks a
# distributed update leaves all actor replicas identical. It writes no weights.
s=s.replace("loss=demonstration_loss(sorted(episodes,key=lambda e:len(e['indices']),reverse=True)[:8])", "selected=sorted(episodes,key=lambda e:len(e['indices']),reverse=True)[rank*8:rank*8+8]\n  for e in selected:\n   for i in e['indices']:\n    with Image.open(rows[i]['path']) as image:all_codes[i].copy_(encode(image.convert('RGB')))\n  loss=demonstration_loss(selected)")
s=s.replace("  if args.arm=='action_rules_only':assert set(gradients)<=set(manifest['trainable_names'])", "  average_gradients(trainable);torch.nn.utils.clip_grad_norm_(trainable,1.);optimizer.step()\n  synced=assert_identical_parameters(model)")
s=s.replace("'optimizer_updates':0,'complete':True", "'optimizer_updates':1,'saved_weights':False,'synchronization_error':synced,'complete':True")
s=s.replace("atomic(out/'verification.json'", "atomic(rank_dir/'verification.json'")
s=s.replace("atomic(out/'startup_verified.json'", "atomic(rank_dir/'startup_verified.json'")
s=s.replace("bank.sample(rng,(update-1)*16+offset+i)", "bank.sample(rng,(update-1)*16*world+rank*16+offset+i)")
s=s.replace("flat=torch.cat([b['advantages'][b['valid']] for b in batches]);mean=flat.mean();std=flat.std(unbiased=False).clamp_min(1e-6)", "flat=torch.cat([b['advantages'][b['valid']] for b in batches]);mean,std=global_moments(flat)")
s=s.replace("   total_actions=sum(int(b['valid'].sum()) for b in batches)", "   total_actions=sum(int(b['valid'].sum()) for b in batches)\n   global_actions=torch.tensor(total_actions,device='cuda');dist.all_reduce(global_actions)\n   rank_weight=world*total_actions/int(global_actions)")
s=s.replace("actor_loss=sum(actor_parts)+.1*anchor;critic_loss=sum(critic_parts)", "actor_loss=rank_weight*sum(actor_parts)+.1*anchor;critic_loss=rank_weight*sum(critic_parts)")
s=s.replace("    norm=torch.nn.utils.clip_grad_norm_(trainable,1.)", "    average_gradients(trainable)\n    norm=torch.nn.utils.clip_grad_norm_(trainable,1.)")
s=s.replace("    critic_loss.backward();value_norm=", "    critic_loss.backward();average_gradients(critic.parameters());value_norm=")
s=s.replace("   save(out/'latest',update)", "   synchronization_error=assert_identical_parameters(model);assert_identical_parameters(critic)\n   save(out/'latest',update)")
s=s.replace("directory=out/'rollouts'/f'step_{update:06d}'", "directory=out/'rollouts'/f'step_{update:06d}'/f'rank{rank}'")
s=s.replace("   record={'update':update", "   global_metrics=[row for part in gather_objects([trial_metric(t) for t in trials]) for row in part]\n   assert len(global_metrics)==16*world\n   record={'rank':rank,'world':world,'synchronization_error':synchronization_error,'global_actions':int(global_actions),'update':update")
s=s.replace("'training':aggregate([trial_metric(t) for t in trials])", "'training':aggregate(global_metrics)")
s=s.replace("append(out/'updates.jsonl',record)", "append(rank_dir/'updates.jsonl',record)")
s=s.replace(" for sim in sims:sim.close()", " for sim in sims:sim.close()\n dist.destroy_process_group()")
s=s.replace("out=base/'symbolic_ppo8_v82';rank_dir=", "out=base/'symbolic_ppo8_v82';out=out/'preflight' if args.verify_only else out;rank_dir=")
s=s.replace("read(out/'SOURCE.json')", "read(base/'symbolic_ppo8_v82/SOURCE.json')")
# The just-completed primary PPO improved development from 35 to 39. Continue
# that successful optimization state rather than discarding its learned rules.
s=s.replace("base/'sequence_perception_v78/frozen/step_000005'", "base/'perceptual_rl_v80/joint/best_development'")
s=s.replace("model.load_state_dict(warm['model']);del warm", "model.load_state_dict(warm['model']);parent_training={k:warm[k] for k in ['optimizer','critic','value_optimizer']}\nparent_epoch=warm['epoch'];del warm")
s=s.replace("value_optimizer=torch.optim.AdamW(critic.parameters(),lr=1e-3,weight_decay=.0001)", "value_optimizer=torch.optim.AdamW(critic.parameters(),lr=1e-3,weight_decay=.0001)\noptimizer.load_state_dict(parent_training['optimizer']);critic.load_state_dict(parent_training['critic']);value_optimizer.load_state_dict(parent_training['value_optimizer']);del parent_training")
s=s.replace("warm_source.parent/'evaluations/step_000005/summary.json'", "warm_source.parent/'evaluations'/f'step_{parent_epoch:06d}'/'summary.json'")
s=s.replace("warm_source.parent/'evaluations/step_000005'", "warm_source.parent/'evaluations'/f'step_{parent_epoch:06d}'")
s=s.replace("'selection':'Development-selected v78 frozen5, 37/80; no fixed80 selection or duplicate baseline rollouts.'", "'selection':'Continue development-selected v80 primary PPO best39/80; descendant of supervised v76 observed_inverse40. No fixed80 selection or duplicate baseline.', 'parent_ppo_update':parent_epoch")
s=s.replace("'optimizer_resume':False", "'optimizer_resume':True")
s=s.replace("'inverse_auxiliary':'Disabled as in the selected supervised source; no action-bearing belief inverse shortcut.'", "'inverse_auxiliary':'Preserve parent observed-ID inverse supervision; no recurrent action-bearing belief inverse shortcut.'")
s=s.replace('from nesy_vla.sequence_perception import SequencePerception', 'from nesy_vla.sequence_perception import SequencePerception\nfrom nesy_vla.observation_inverse import observation_inverse_logits')
s=s.replace("    'visual_last_block':perception.block.state_dict()", "    'parent_source':str(warm_source)")
front_start=s.index('    prefix=torch.nn.Sequential',s.index('def initialize_evaluation'))
front_end=s.index('    # Re-encode training data',front_start)
s=s[:front_start]+s[front_end:]
code_start=s.index('    with torch.autocast',s.index('def encode(image):'))
code_end=s.index('\n@torch.no_grad()\ndef evaluate',code_start)
s=s[:code_start]+'''    with torch.autocast('cuda',dtype=torch.bfloat16):
        f=F.adaptive_avg_pool2d(vision(image_tensor(image,vision_cfg['image_size'],'cuda')),(7,7)).flatten(2).transpose(1,2)
    z=F.normalize(((f.float()-vocabulary['mean'])/vocabulary['sd'])@vocabulary['axes'],dim=-1)
    return torch.cdist(z.flatten(0,1),vocabulary['centers']).argmin(1)
'''+s[code_end:]
s=s.replace("        for distribution in [r['observed_probabilities'],r['state_probabilities']]:", "        if step:\n            inverse=observation_inverse_logits(model,program,x[:,step-1],x[:,step],step)\n            inverse_loss=(F.cross_entropy(inverse,y[:,step-1],reduction='none')*active).sum()/denominator\n            loss=loss+.1*inverse_loss\n        for distribution in [r['observed_probabilities'],r['state_probabilities']]:")
front_start=s.index('    # Re-encode training data',s.index('def initialize_evaluation'))
front_end=s.index('    if launch_simulators:',front_start)
s=s[:front_start]+'''    # The v80 parent kept the identical v73 frontend; reuse the existing,
    # hash-verified integer corpus instead of recomputing 24,848 images.
    for i in [0,len(rows)//2,len(rows)-1]:
        with Image.open(rows[i]['path']) as image:assert torch.equal(encode(image.convert('RGB')),all_codes[i])
'''+s[front_end:]
compile(s,'train_symbolic_ppo8_v82.py','exec')
(root/'scripts/train_symbolic_ppo8_v82.py').write_text(s,encoding='utf-8')
print('Generated one synchronous PPO experiment; 128 training rollouts per global update on 8 GPUs.')
