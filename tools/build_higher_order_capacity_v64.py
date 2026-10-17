from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'scripts/train_probabilistic_automaton_v56.py').read_text()
s=s.replace('from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton','from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton\nfrom nesy_vla.higher_order_automaton import HigherOrderAutomaton')
s=s.replace("choices=['soft_recurrent','soft_reset']","choices=['pairwise','higher_order']").replace("default=60)","default=20)").replace("default=20);args=", "default=10);args=")
s=s.replace("out=base/'probabilistic_automaton_v56'", "out=base/'higher_order_capacity_v64'")
start=s.index('architecture={');end=s.index('state={',start)
s=s[:start]+r'''
warm_source=base/'probabilistic_automaton_v56/soft_recurrent/step_000040';warm=load_committed(warm_source)
architecture=dict(warm['manifest']['architecture']);assert not architecture['reset_memory']
assert warm['manifest']['data']==bank.provenance
reference=ProbabilisticTokenAutomaton(**architecture).cuda().eval();reference.load_state_dict(warm['model'])
if args.arm=='higher_order':
 architecture['higher_rules']=128;model=HigherOrderAutomaton(**architecture).cuda();model.initialize_pairwise(warm['model'])
else:
 model=ProbabilisticTokenAutomaton(**architecture).cuda();model.load_state_dict(warm['model'])
for name,param in model.named_parameters():
 param.requires_grad_(name in ['conclusions','conclusion_basis','weight_logits','weight_basis'] or name.startswith('higher_'))
with torch.no_grad():
 x=torch.tensor(codes[:8].astype(np.int64),device='cuda');task=torch.arange(8,device='cuda')%2
 _,memory=model.initial_state(8,'cuda');a=torch.full((8,),17,device='cuda',dtype=torch.long)
 before=reference.step(reference.generate(task,x[:,None],0),x,memory,a,0)
 after=model.step(model.generate(task,x[:,None],0),x,memory,a,0)
 assert torch.equal(before['observed_ids'],after['observed_ids']) and torch.equal(before['state_values'],after['state_values'])
 initial_error=float((before['probabilities']-after['probabilities']).abs().max())
 if args.arm=='pairwise':assert initial_error==0
del reference,warm;gc.collect();torch.cuda.empty_cache()
optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=.0001)
manifest={'architecture':architecture,'arm':args.arm,'warm_start_checkpoint':str(warm_source),
 'warm_start_commit':read(warm_source/'COMMITTED'),'initial_action_probability_difference':initial_error,
 'source_commit':source_commit,'data':bank.provenance,'train_rows_sha256':sha(source/'training_rows.json'),
 'recovery_tokens':extra_commit,'epochs':args.epochs,'batch_size':args.batch_size,'evaluate_every':args.evaluate_every,
 'parameters':sum(p.numel() for p in model.parameters()),'horizon':32,'revision_interval':10,'fixed80_used':False,
 'trainable_names':[n for n,p in model.named_parameters() if p.requires_grad],
 'comparison':'Frozen identical learned observation/state encoders and program coefficients fromv56recurrent40. Both learn existing action rules; treatment adds128 generated soft conjunctions over up to4anonymous categorical state factors. Initial higher-order mixture10percent.',
 'inference':'All action distributions obtained by explicit finite-state rule sums. No maps, manually named predicates or visual-feature action head.',
 'source_development':read(warm_source.parent/'evaluations/step_000040/summary.json')}
if (out/'manifest.json').exists():assert read(out/'manifest.json')==manifest
else:atomic(out/'manifest.json',manifest)
''' + s[end:]
# The original trainer's exact data checks, supervision masks and closed-loop
# evaluator are retained. Save extended rule evidence for visual inspection.
s=s.replace("'program_start':program.start,'image_sha256':", "'higher_order_rule_support':proof.get('higher_order',{}).get('normalized_rule_support',torch.empty(n,0,device='cuda'))[i].tolist(),'program_start':program.start,'image_sha256':")
s=s.replace("'Soft beliefs over finite learned state IDs; exact transition marginalization and product-conjunction action rules.'", "'Soft beliefs over finite learned state IDs; exact transition marginalization and generated pairwise or higher-order product-conjunction rules.'")
(root/'scripts/train_higher_order_capacity_v64.py').write_text(s)
