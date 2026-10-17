from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
text=(root/'preflight_raw_boundary_v19.py').read_text(encoding='utf-8')
text=text.replace('from predictive_nesy.raw_boundary_data import RawBoundaryFeatures','from predictive_nesy.replayed_boundary_data import ReplayedBoundaryFeatures')
text=text.replace('preflight_raw_boundary_v19','preflight_replayed_boundary_v20')
text=text.replace('from train_raw_boundary_v19 import TrainingModule','from train_replayed_boundary_v20 import TrainingModule')
text=text.replace("features=RawBoundaryFeatures(root/'prequant_visual_features_v18','cuda')", "features=ReplayedBoundaryFeatures(root/'prequant_visual_features_v18',root/'replayed_boundary_features_v20','cuda')")
text=text.replace('features.transitions(data)','features.transitions(data,bank.directory)')
text=text.replace('available_fraction=float(data[\'boundary_available\'].float().mean())',
    "available_fraction=float(data['boundary_available'].float().mean());assert available_fraction==1.")
text=text.replace('source_sha256=sha(source),time=time.time()',"source_sha256=sha(source),replay_manifest_sha256=sha(root/'replayed_boundary_features_v20/manifest.json'),time=time.time()")
(root/'preflight_replayed_boundary_v20.py').write_text(text,encoding='utf-8')
