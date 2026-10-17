from pathlib import Path
s=Path('NeSy-VLA/scripts/train_perceptual_fsm_v74.py').read_text(encoding='utf-8')
def replace(a,b):
 global s
 assert s.count(a)==1,a
 s=s.replace(a,b)
replace("choices=['frozen','tail_1e4']","choices=['observed_inverse','no_inverse']")
replace("out=base/'perceptual_fsm_v74'/args.arm","out=base/'observation_inverse_v76'/args.arm")
replace("reservation_root=base/'perceptual_fsm_v74'","reservation_root=base/'observation_inverse_v76'")
replace("source=base/'perceptual_sequences_v73'/args.arm","source=base/'perceptual_sequences_v73'/'tail_1e4'")
replace("warm_source=base/'perceptual_learning_v72'/args.arm/'best_audit'","warm_source=base/'perceptual_learning_v72'/'tail_1e4'/'best_audit'")
replace("from nesy_vla.small_checkpoint import save_small_checkpoint","from nesy_vla.small_checkpoint import save_small_checkpoint\nfrom nesy_vla.observation_inverse import observation_inverse_logits")
replace("'inverse_action':.1","'inverse_action':.1 if args.arm=='observed_inverse' else 0.")
replace("'comparison':'Matched fullsequence data/objectives/epochs/optimizer/schedule. Initialweights are each arms best training-onlycomponent model (control100,learned600), not equal warmtrainingsteps.',","'comparison':'Identical v74 learned-tail source model, codes, RNG, optimizer, batches, epochs and losses, except inverse input changed to hard observation equalities or inverse loss disabled. Existing v74tail is the original recurrent-belief inverse control; do not rerun it.',\n 'inverse_arm':args.arm,\n 'inverse_target_boundary':'Executed action is a label only. Ground both observations with same current generated program; no action-bearing memory enters the inverse head.',")
replace("'perception':'Frozen afterv72componenttraining; canonicalvisualbatch1. OriginalCNN control versus learnedCNNtail. Both use exactv53vocabulary.'","'perception':'Same fixed v72learnedtail as livev74tail1e4; exactv53vocabulary andcanonicalbatch1.'")
replace('        if before_state is not None:\n            inverse=model.inverse_action(before_state,r[\'state_values\'])',"        if step and args.arm=='observed_inverse':\n            inverse=observation_inverse_logits(model,program,x[:,step-1],x[:,step],step)")
replace("    if args.arm!='frozen':","    if True:")
compile(s,'train_observation_inverse_v76.py','exec')
Path('NeSy-VLA/scripts/train_observation_inverse_v76.py').write_text(s,encoding='utf-8')
print('Built action-label-isolated inverse supervision with matched no-inverse control.')
