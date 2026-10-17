import hashlib,json,os,shutil,time
from pathlib import Path
base=Path('/home/user/xjy/nesy_vla_20261010');root=base/'symbolic_vision_v8';source=base/'frozen_vision_v8_code'
assert '48 passed' in (source/'verification.log').read_text()
proof=json.loads((source/'pretrained_vision_verification.json').read_text())
assert proof['all_actions_via_discrete_symbols_and_generated_rules'] and proof['gradients']['grounder.vision.']==0
assert not root.exists(),'An existing experiment must be inspected, not overwritten'
root.mkdir()
for name in ['nesy_vla','scripts','tests']:shutil.copytree(source/name,root/name,ignore=shutil.ignore_patterns('__pycache__'))
(root/'configs').mkdir();origin=base/'symbolic_robust_v7/initialization_source';target=root/'initialization_source';target.mkdir()
marker=json.loads((origin/'COMMITTED').read_text());checkpoint=origin/marker['checkpoint_file'];os.link(checkpoint,target/checkpoint.name)
assert hashlib.sha256((target/checkpoint.name).read_bytes()).hexdigest()==marker['checkpoint_sha256']
(target/'COMMITTED').write_text(json.dumps(marker,indent=2))
cfg=json.loads((base/'symbolic_robust_v7/configs/robust_multieffect.json').read_text());cfg.pop('initialize_from')
cfg.update(root=str(root),initialize_symbolic_from=str(target),freeze_vision=True,vision_cache_entries=4096,
    demonstrations_per_rank=144,training_chunk_batch=64,maximum_chunk_batch=128,training_frame_budget=768,
    revision_chunk_limit=64,rollouts_per_rank=6,vision_microbatch=8)
for name,backbone in [('frozen_convnext','convnext_tiny'),('frozen_qwen','qwen35_frozen')]:
    arm=json.loads(json.dumps(cfg));arm.update(architecture=name+'_v8',vision_backbone=backbone)
    (root/'configs'/f'{name}.json').write_text(json.dumps(arm,indent=2))
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ['nesy_vla','scripts','tests','configs'] for p in (root/name).rglob('*') if p.is_file()}
record={'time':time.time(),'initialization_source':marker,'initialization':'Only generator and symbolic interpreter restored; fresh learned grounder and optimizer in both arms.',
    'comparison':'Frozen ImageNet ConvNeXt visual evidence vs frozen Qwen3.5 visual evidence. Projection input widths differ; shared learned-grounding/rule architecture and data.',
    'trainable_qwen_arm_parameters':proof['trainable_parameters'],'frozen_qwen_visual_parameters':proof['vision_parameters'],
    'qwen_language_model_loaded':False,'action_decoder_loaded':False,'maps':False,'human_predicate_labels':False,
    'horizon':32,'revision_interval':10,'fixed_validation':'Original fixed80 and development80 unchanged, every10 updates',
    'data':'Same17560 raw lossless training frames, with photometric augmentation and public collision feedback',
    'feature_cache':'RAM only; hash actual image pixels; frozen features; shared with EMA teacher; no policy outputs cached',
    'tests_passed':48,'navigation_success_unproven':True,'source_files':manifest}
(root/'experiment.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k!='source_files'},indent=2))
