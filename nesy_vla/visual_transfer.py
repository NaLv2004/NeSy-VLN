"""Transfer only convolutional evidence features; never copy action/cost heads."""
import torch


def single_image_features(state):
    prefix='encoder.features.'
    selected={k[len(prefix):]:v.clone() for k,v in state.items() if k.startswith(prefix)}
    if not selected:raise ValueError('No convolutional feature weights in source')
    key='0.0.weight';weight=selected[key]
    if weight.shape[1] not in [3,9]:raise ValueError('Expected one or three RGB frames in source')
    if weight.shape[1]==9:
        # Exact first-layer response for three identical RGB observations.
        selected[key]=weight.reshape(weight.shape[0],3,3,*weight.shape[2:]).sum(1)
    return selected


def initialize_visual_evidence(folder,model):
    from stripsnav.io import load_committed,read_json
    from pathlib import Path
    payload=load_committed(folder)
    selected=single_image_features(payload['model'])
    model.grounder.vision.load_state_dict(selected,strict=True)
    model.teacher.vision.load_state_dict(selected,strict=True)
    channels=payload['model']['encoder.features.0.0.weight'].shape[1]
    return {'source':str(folder),'source_commit':read_json(Path(folder)/'COMMITTED'),
            'copied_keys':sorted(selected),'copied_prefix':'encoder.features.',
            'action_or_cost_heads_copied':False,
            'input_adaptation':'Unchanged single RGB input' if channels==3 else 'Sum temporal RGB kernel groups for repeated RGB input',
            'source_input_channels':channels,'source_temporal_history_preserved':channels==3}
