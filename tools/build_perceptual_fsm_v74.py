from pathlib import Path
source=Path('NeSy-VLA/scripts/train_probabilistic_automaton_v56.py').read_text(encoding='utf-8')
def replace(old,new):
 global source
 assert source.count(old)==1,old
 source=source.replace(old,new)
replace("choices=['soft_recurrent','soft_reset']","choices=['frozen','tail_1e4']")
replace("default=20);args=p.parse_args()","default=10);args=p.parse_args()")
replace("args=p.parse_args()", "p.add_argument('--verify-only',action='store_true');args=p.parse_args()")
replace("out=base/'probabilistic_automaton_v56'/args.arm","out=base/'perceptual_fsm_v74'/args.arm")
replace("from stripsnav.io import committed_torch_save,load_committed",'''from stripsnav.io import load_committed
from nesy_vla.small_checkpoint import save_small_checkpoint
def committed_torch_save(folder,payload,metadata):
    return save_small_checkpoint(folder,payload,metadata,reservation_root=base/'perceptual_fsm_v74')''')
begin=source.index("source=base/'spatial_automaton_v43'")
end=source.index('assert codes.shape==',begin)
source=source[:begin]+'''source=base/'perceptual_sequences_v73'/args.arm;extra=base/'data/onpolicy_recovery_v48'
source_commit=read(source/'TOKENS_COMMITTED')
assert bank.provenance==source_commit['data_provenance']
assert sha(source/'visual_tokens.npy')==source_commit['tokens_sha256']
assert sha(source/'training_rows.json')==source_commit['training_rows_sha256']
assert sha(source_commit['vocabulary_path'])==source_commit['vocabulary_sha256']
rows=read(source/'training_rows.json');old_count=source_commit['old_count']
codes=np.load(source/'visual_tokens.npy')
extra_commit={'training_rows_sha256':source_commit['recovery_rows_sha256']}
'''+source[end:]
replace("'reset_memory':args.arm=='soft_reset'","'reset_memory':False")
begin=source.index("warm_source=base/'factorized_automaton_v51")
end=source.index('optimizer=torch.optim.AdamW',begin)
source=source[:begin]+'''warm_source=base/'perceptual_learning_v72'/args.arm/'best_audit'
assert read(warm_source/'COMMITTED')==source_commit['component_commit']
warm=load_committed(warm_source)
assert architecture==warm['manifest']['architecture']
model.load_state_dict(warm['model'],strict=True)
del warm;gc.collect();torch.cuda.empty_cache()
'''+source[end:]
begin=source.index("manifest={'architecture':")
end=source.index("if (out/'manifest.json')",begin)
source=source[:begin]+'''manifest={'architecture':architecture,'warm_start_checkpoint':str(warm_source),'warm_start_commit':read(warm_source/'COMMITTED'),
 'optimizer_resume':False,'source_commit':source_commit,'recovery_commit':read(extra/'COMPLETE'),
 'train_rows_sha256':sha(source/'training_rows.json'),'train_tokens_sha256':sha(source/'visual_tokens.npy'),'data':bank.provenance,
 'epochs':args.epochs,'batch_size':args.batch_size,'evaluate_every':args.evaluate_every,'parameters':sum(p.numel() for p in model.parameters()),
 'horizon':32,'revision_interval':10,'training_inputs':'Only hard integer patch codes, task IDs and demonstrated executed actions.',
 'learned_program':'Small history model generates learned predicate definitions, soft transition tables and action conclusions every10 actions.',
 'rule_semantics':'Conjunctions of equalities over anonymous automatically learned observation/state IDs. No human predicate semantics.',
 'losses':{'action':1.,'observation_reconstruction':.2,'next_discrete_observation':.2,'inverse_action':.1,'state_balance':.005},
 'fixed80_used':False,'oracle_or_geometry_at_inference':False,'recovery_supervised_weight':4.,'context_prefix_imitation_weight':0.,
 'teacher_forced_training_accuracy_is_not_navigation_success':True,
 'state_semantics':'Hard observed predicate IDs, explicit soft beliefs over finite learned state IDs, exact rule-based transition marginalization.',
 'perception':'Frozen afterv72componenttraining; canonicalvisualbatch1. OriginalCNN control versus learnedCNNtail. Both use exactv53vocabulary.',
 'comparison':'Matched fullsequence data/objectives/epochs/optimizer/schedule. Initialweights are each arms best training-onlycomponent model (control100,learned600), not equal warmtrainingsteps.',
 'checkpoint_policy':'Complete model/optimizer/RNG atomic commits with512MiB+2xactualserializedsize reservation andshared experiment lock.'}
'''+source[end:]
old="""    vision=visual.grounder.vision;del visual;gc.collect()
    vocabulary=torch.load(source/'visual_vocabulary.pt',map_location='cuda',weights_only=True)
    for p in sorted((source/'development').glob('*/trace.json'))[:3]:
        old=read(p);assert encode(Image.open(p.parent/'first.png').convert('RGB')).tolist()==old['steps'][0]['symbols'][:49]
"""
new="""    vision=visual.grounder.vision;del visual
    if hasattr(vision,'backbone'):vision=vision.backbone
    assert isinstance(vision,torch.nn.Sequential) and len(vision)==8
    assert read(loc/'COMMITTED')==source_commit['encoder_source_commit']
    if args.arm!='frozen':
        perception=load_committed(warm_source);vision[6:].load_state_dict(perception['visual_tail']);del perception
    gc.collect()
    vocabulary=torch.load(source_commit['vocabulary_path'],map_location='cuda',weights_only=True)
    for index in [0,len(rows)//2,len(rows)-1]:
        assert encode(Image.open(rows[index]['path']).convert('RGB')).tolist()==codes[index].tolist(),'Training/deployment visual token mismatch'
    atomic(out/'perception_replay_verified.json',{'canonical_visual_batch':1,'checked_training_rows':[0,len(rows)//2,len(rows)-1],
        'exact_integer_token_match':True,'encoder_commit':source_commit['component_commit'],'vocabulary_sha256':source_commit['vocabulary_sha256']})
"""
replace(old,new)
replace('def initialize_evaluation():','def initialize_evaluation(launch_simulators=True):')
replace("    sims=[bank.simulator(out/f'sim_{j}.log',0,evaluation=True) for j in range(8)]", "    if not launch_simulators:return\n    sims=[bank.simulator(out/f'sim_{j}.log',0,evaluation=True) for j in range(8)]")
replace('try:\n    if begin', '''try:
    if args.verify_only:
        initialize_evaluation(launch_simulators=False)
        selected=sorted(episodes,key=lambda e:len(e['indices']),reverse=True)[:2]
        x=all_codes[torch.tensor([e['indices'][0] for e in selected],device='cuda')]
        task=torch.tensor([e['task'] for e in selected],device='cuda')
        _,prior=model.initial_state(len(x),'cuda')
        previous_action=torch.full((len(x),),17,device='cuda',dtype=torch.long)
        program=model.generate(task,x[:,None],0)
        result=model.step(program,x,prior,previous_action,0)
        proof=model.proof(program,result['observed_ids'],prior,previous_action,result['state_values'],0)
        proof_error=float((proof['probabilities']-result['probabilities']).abs().max())
        assert proof_error==0
        del program,result,proof
        score=train_batch(selected)
        assert score['gradient_norm']>0 and score['supervised']>0
        atomic(out/'integration_verified.json',{'time':time.time(),'arm':args.arm,
            'actual_training_episode_ids':[e['id'] for e in selected],
            'episode_lengths':[len(e['indices']) for e in selected],
            'gradient_norm':score['gradient_norm'],'loss':score['loss'],
            'integer_action_proof_max_error':proof_error,
            'perception_token_equality':True,'optimizer_step_completed':True,
            'checkpoint_saved':False,'navigation_evaluation_run':False})
        print(json.dumps(read(out/'integration_verified.json')),flush=True)
        raise SystemExit(0)
    if begin''')
replace("except BaseException as error:progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise", "except Exception as error:progress(state='failed',error=repr(error),traceback=traceback.format_exc());raise")
compile(source,'train_perceptual_fsm_v74.py','exec')
Path('NeSy-VLA/scripts/train_perceptual_fsm_v74.py').write_text(source,encoding='utf-8')
print('Built full-sequence generated-state-machine experiment with matched learned/frozen perception.')
