import hashlib,json,os,shutil,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_robust_v7'
root.mkdir(exist_ok=True)
source=base/'robust_grounding_v7_code'
for name in ['nesy_vla','scripts','tests']:
    if (root/name).exists():raise RuntimeError('Experiment code already present; do not overwrite')
    shutil.copytree(source/name,root/name,ignore=shutil.ignore_patterns('__pycache__'))
(root/'configs').mkdir()
origin=base/'symbolic_sft_v5/runs/sft_plain/step_000050'
marker=json.loads((origin/'COMMITTED').read_text())
assert marker['optimizer_included'] and marker['epoch']==50
target=root/'initialization_source';target.mkdir();checkpoint=origin/marker['checkpoint_file']
os.link(checkpoint,target/checkpoint.name)
assert hashlib.sha256((target/checkpoint.name).read_bytes()).hexdigest()==marker['checkpoint_sha256']
(target/'COMMITTED').write_text(json.dumps(marker,indent=2))
cfg=json.loads((base/'symbolic_sft_v5/configs/sft_plain.json').read_text())
dataset=base/'data/lossless_demonstrations_v7'
cfg.update(root=str(root),initialize_from=str(target),robust_grounding=True,balanced_transition_loss=True,
    demonstration_file=str(dataset/'examples.json'),demonstration_image_root=str(dataset),
    lossless_trace_images=True,training_frame_budget=220,revision_chunk_limit=28)
cfg['loss'].update(visual_consistency=5.,quantization=.05)
for name,multi in [('robust_reference',False),('robust_multieffect',True)]:
    new=json.loads(json.dumps(cfg));new['architecture']=name+'_v7';new['program']['multi_effects']=multi
    (root/'configs'/f'{name}.json').write_text(json.dumps(new,indent=2))
record={'time':time.time(),'initialization':marker,'origin':str(origin),'training_mode':'supervised',
    'comparison':'Only program.multi_effects differs; all model/optimizer tensors initialize identically.',
    'shared_changes':['Lossless online RGB and real public feedback in existing successful demonstrations',
        'Geometry-preserving photometric and codec augmentation','Clean/augmented predicate consistency',
        'Small predicate margin regularizer','Balanced changed/stable transition loss'],
    'new_effect_representation':'Independent unchanged/add/delete choice per automatically learned predicate',
    'no_direct_image_action_branch':True,'no_handwritten_predicates':True,'no_maps':True,
    'horizon':32,'revision_interval':10,'fixed_validation_unchanged':True,'evaluation_every':10,
    'qualification':'No navigation improvement claimed until actual closed-loop evaluations complete.'}
(root/'experiment.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
