"""Build matched whole-trajectory perceptual learning from the audited trainer."""
from pathlib import Path
s=Path('NeSy-VLA/scripts/train_observation_inverse_v76.py').read_text(encoding='utf-8')
def replace(old,new):
 global s
 assert s.count(old)==1,(old,s.count(old))
 s=s.replace(old,new)
replace("choices=['observed_inverse','no_inverse']","choices=['frozen','joint']")
replace("default=60);p.add_argument('--batch-size',type=int,default=64)","default=10);p.add_argument('--batch-size',type=int,default=32)")
replace("default=10);p.add_argument('--verify-only'","default=5);p.add_argument('--verify-only'")
s=s.replace("base/'observation_inverse_v76'","base/'sequence_perception_v78'")
replace('from nesy_vla.observation_inverse import observation_inverse_logits',
'''from nesy_vla.learned_visual_tokens import TrainableAlphabetAutomaton, SymbolTokens
from nesy_vla.sequence_perception import SequencePerception
from nesy_vla.model import NavigationModel''')
replace('model=ProbabilisticTokenAutomaton(**architecture).cuda()', 'model=TrainableAlphabetAutomaton(**architecture).cuda()')
replace("warm_source=base/'perceptual_learning_v72'/'tail_1e4'/'best_audit'", "warm_source=base/'perceptual_fsm_v74'/'tail_1e4'/'step_000010'")
replace("assert read(warm_source/'COMMITTED')==source_commit['component_commit']", "assert read(warm_source/'COMMITTED')['checkpoint_sha256']=='039086a4e972c7016fce81e226c6c42c71a3b3fd697b9dad3f045917daad71cd'")
replace("optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.0001)", '''location=Path(source_commit['encoder_source']);vision_cfg=read(location.parent/'config.json')
assert read(location/'COMMITTED')==source_commit['encoder_source_commit']
visual=NavigationModel(vision_cfg).eval();load_checkpoint(location,visual);visual.requires_grad_(False).cuda()
encoder=visual.grounder.vision;del visual
if hasattr(encoder,'backbone'):encoder=encoder.backbone
assert isinstance(encoder,torch.nn.Sequential) and len(encoder)==8
component=Path(source_commit['component_checkpoint']);assert read(component/'COMMITTED')==source_commit['component_commit']
perception_payload=load_committed(component);encoder[6:].load_state_dict(perception_payload['visual_tail']);del perception_payload
prefix=torch.nn.Sequential(*list(encoder.children())[:7],torch.nn.Sequential(*list(encoder[7].children())[:-1])).eval().requires_grad_(False)
vocabulary=torch.load(source_commit['vocabulary_path'],map_location='cuda',weights_only=True)
perception=SequencePerception(encoder[7][-1],vocabulary,canonical_batch=16).cuda()
perception.block.requires_grad_(args.arm=='joint');del encoder;gc.collect()
groups=[{'params':model.parameters(),'lr':1e-4}]
if args.arm=='joint':groups.append({'params':perception.block.parameters(),'lr':1e-5})
optimizer=torch.optim.AdamW(groups,weight_decay=.0001)''')
replace("'inverse_action':.1 if args.arm=='observed_inverse' else 0.","'inverse_action':0.")
replace("'perception':'Same fixed v72learnedtail as livev74tail1e4; exactv53vocabulary andcanonicalbatch1.'", "'perception':'Matched canonical last-block batch16 in both arms. Frozen prefix cached only in private GPU RAM. Joint arm trains final ConvNeXt block through hard tokens over complete trajectories.'")
replace("'comparison':'Identical v74 learned-tail source model, codes, RNG, optimizer, batches, epochs and losses, except inverse input changed to hard observation equalities or inverse loss disabled. Existing v74tail is the original recurrent-belief inverse control; do not rerun it.',", "'comparison':'Same immutable v74tail epoch10 model, v72 visual tail, seed, original training rows, batch32 and AdamW1e-4. Frozen control versus final-visual-block learning at1e-5 through hard tokens. Both inverse auxiliaries disabled. No baseline navigation rerun.',\n 'canonical_last_block_batch':16,'model_lr':1e-4,'visual_lr':1e-5 if args.arm=='joint' else 0.,")
replace("'inverse_arm':args.arm,\n 'inverse_target_boundary':'Executed action is a label only. Ground both observations with same current generated program; no action-bearing memory enters the inverse head.',", "'inverse_arm':'disabled','visual_parameters':sum(p.numel() for p in perception.block.parameters()),")
replace("model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);begin=saved['epoch']", "model.load_state_dict(saved['model']);perception.block.load_state_dict(saved['visual_last_block']);optimizer.load_state_dict(saved['optimizer']);begin=saved['epoch']")
replace("payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'manifest':manifest,", "payload={'model':model.state_dict(),'visual_last_block':perception.block.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'manifest':manifest,")
anchor='\ndef train_batch(batch):'
replace(anchor, '''
# Frozen features are confined to this visual frontend. No feature tensor is
# handed to the state-machine generator, transition interpreter or action rules.
prefix_cache=torch.empty((len(rows),768,7,7),device='cuda',dtype=torch.bfloat16)
cached=np.zeros(len(rows),dtype=bool)
@torch.no_grad()
def cache_prefix(indices):
    for i in indices:
        i=int(i)
        if cached[i]:continue
        with Image.open(rows[i]['path']) as image:pixel=image_tensor(image.convert('RGB'),vision_cfg['image_size'],'cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):f=prefix(pixel)
        assert f.shape==(1,768,7,7)
        prefix_cache[i].copy_(f[0]);cached[i]=True
        if int(cached.sum())%1024==0:progress(state='caching_frozen_prefix',completed=int(cached.sum()),total=len(rows))

def tokens_for_indices(indices):
    unique,inverse=torch.unique(indices.flatten(),sorted=True,return_inverse=True)
    cache_prefix(unique.tolist())
    tokens=perception(prefix_cache[unique])
    return SymbolTokens(tokens.ids[inverse].reshape(*indices.shape,49),tokens.values[inverse].reshape(*indices.shape,49,64))

def train_batch(batch):''')
replace("x=all_codes[indices];y=labels[indices];mask=valid&supervised[indices]", "x=tokens_for_indices(indices);y=labels[indices];mask=valid&supervised[indices]")
replace("total=x.new_zeros((),dtype=torch.float)","total=x.ids.new_zeros((),dtype=torch.float)")
replace("        reconstruction=F.cross_entropy(r['reconstruction'].transpose(1,2),x[:,step],reduction='none').mean(-1)", "        reconstruction=F.cross_entropy(r['reconstruction'].transpose(1,2),x.ids[:,step],reduction='none').mean(-1)")
replace("prediction.transpose(1,2),x[:,step+1],reduction='none'", "prediction.transpose(1,2),x.ids[:,step+1],reduction='none'")
replace("prediction.argmax(-1)==x[:,step+1]", "prediction.argmax(-1)==x.ids[:,step+1]")
replace("        if step and args.arm=='observed_inverse':\n            inverse=observation_inverse_logits(model,program,x[:,step-1],x[:,step],step)\n            inverse_loss=(F.cross_entropy(inverse,y[:,step-1],reduction='none')*active).sum()/denominator\n            loss=loss+.1*inverse_loss;counts['inverse_loss']+=float(inverse_loss.detach())", '''        if args.verify_only:
            with torch.no_grad():
                plain_program=model.generate(task,x.ids[:,max(0,program.start-10):program.start+1],program.start)
                plain=model.step(plain_program,x.ids[:,step],previous.detach(),last_action,step)
                assert torch.equal(plain['probabilities'],r['probabilities'])
                proof=model.proof(program,r['observed_ids'],previous,last_action,r['state_values'],step)
                assert torch.equal(proof['probabilities'],r['probabilities'])''')
replace("norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm);optimizer.step()", "norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm)\n    visual_norm=torch.nn.utils.clip_grad_norm_(perception.block.parameters(),1.) if args.arm=='joint' else torch.tensor(0.)\n    assert torch.isfinite(visual_norm)\n    if args.arm=='joint':assert visual_norm>0,'No visual gradient through sequence rules'\n    optimizer.step()")
replace("counts.update(loss=float(total.detach()),gradient_norm=float(norm),state_counts=seen.cpu().tolist())", "counts.update(loss=float(total.detach()),gradient_norm=float(norm),visual_gradient_norm=float(visual_norm),state_counts=seen.cpu().tolist())")
replace("x=all_codes[torch.tensor(inputs,device='cuda')];goal=", "x=tokens_for_indices(torch.tensor(inputs,device='cuda'));goal=")
start=s.index('vision=None;sims=[];vision_cfg=None;vocabulary=None')
end=s.index('@torch.no_grad()\ndef evaluate(epoch):',start)
s=s[:start]+'''sims=[]
@torch.no_grad()
def encode(image):
    with torch.autocast('cuda',dtype=torch.bfloat16):features=prefix(image_tensor(image,vision_cfg['image_size'],'cuda'))
    return perception(features).ids[0]

def initialize_evaluation(launch_simulators=True):
    for index in [0,len(rows)//2,len(rows)-1]:
        cache_prefix([index])
        with Image.open(rows[index]['path']) as image:online=encode(image.convert('RGB'))
        offline=tokens_for_indices(torch.tensor([index],device='cuda')).ids[0]
        assert torch.equal(online,offline),'Cached and online perception disagree'
    atomic(out/'perception_replay_verified.json',{'canonical_visual_prefix_batch':1,'canonical_last_block_batch':16,
        'training_rows_checked':3,'exact_integer_token_match':True,'features_stay_inside_perception':True})
    if launch_simulators and not sims:
        sims.extend(bank.simulator(out/f'sim_{j}.log',0,evaluation=True) for j in range(8))
        atomic(out/'processes.json',{'pid':os.getpid(),'simulator_pids':[s.process.pid for s in sims],'time':time.time()})

'''+s[end:]
replace("x=all_codes[torch.tensor([e['indices'][0] for e in selected],device='cuda')]", "x=tokens_for_indices(torch.tensor([e['indices'][0] for e in selected],device='cuda'))")
replace("'gradient_norm':score['gradient_norm'],'loss':score['loss'],", "'gradient_norm':score['gradient_norm'],'visual_gradient_norm':score['visual_gradient_norm'],'loss':score['loss'],\n            'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,")
replace("    if begin and begin%args.evaluate_every==0", "    cache_prefix(range(len(rows)))\n    if begin and begin%args.evaluate_every==0")
replace("'gradient_norm_mean':float(np.mean([r['gradient_norm'] for r in scores])),", "'gradient_norm_mean':float(np.mean([r['gradient_norm'] for r in scores])),\n         'visual_gradient_norm_mean':float(np.mean([r['visual_gradient_norm'] for r in scores])),")
compile(s,'train_sequence_perception_v78.py','exec')
Path('NeSy-VLA/scripts/train_sequence_perception_v78.py').write_text(s,encoding='utf-8')
print('Built full-sequence final-block learning and matched frozen control.')
