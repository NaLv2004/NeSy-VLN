"""Create a new trainer while leaving v19's measured source intact."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
text=(root/'train_raw_boundary_v19.py').read_text(encoding='utf-8')
def replace(a,b):
    global text
    if text.count(a)!=1:raise ValueError((a,text.count(a)))
    text=text.replace(a,b)
replace('from predictive_nesy.raw_boundary_data import RawBoundaryFeatures',
        'from predictive_nesy.replayed_boundary_data import ReplayedBoundaryFeatures')
replace("features=RawBoundaryFeatures(BASE/'predictive_rules_20261015/prequant_visual_features_v18',device)",
        "features=ReplayedBoundaryFeatures(BASE/'predictive_rules_20261015/prequant_visual_features_v18',BASE/'predictive_rules_20261015/replayed_boundary_features_v20',device)")
replace('features.transitions(data)','features.transitions(data,source.directory)')
replace('raw_boundary_input=features.manifest,','raw_boundary_input=features.manifest,raw_transition_boundary_replay=features.replay_manifest,')
replace("world_model_intervention='Raw CNN/PCA boundary evidence before patch quantization conditions complete generated rules and a discrete subgoal; runtime remains symbolic'",
        "world_model_intervention='Canonical raw visual histories recovered for each training transition bank; complete rules/subgoal and symbolic runtime otherwise unchanged'")
(root/'train_replayed_boundary_v20.py').write_text(text,encoding='utf-8')
print('train_replayed_boundary_v20.py generated')
