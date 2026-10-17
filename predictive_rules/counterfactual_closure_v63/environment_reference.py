"""Reuses frozen splits and perceptual assets; policy and simulator stay apart."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .data import BASE, read, sha


def dependencies():
    # Exact source snapshot used by the measured 1097 checkpoint.
    sys.path.insert(0, str(BASE / 'original_ppo_resume_v99/src'))
    from nesy_vla.runtime import setup
    cfg = read(BASE / 'symbolic_execution_v38/configs/grounded_outcome_feedback.json')
    setup(cfg)
    return cfg


def load_previous():
    dependencies()
    from stripsnav.io import load_committed
    from nesy_vla.probabilistic_automaton import ProbabilisticTokenAutomaton
    checkpoint = BASE / 'original_ppo_resume_v99/latest'
    commit = read(checkpoint / 'COMMITTED')
    payload = load_committed(checkpoint)
    if payload['epoch'] != 1097:
        raise RuntimeError('Comparison checkpoint unexpectedly changed')
    model = ProbabilisticTokenAutomaton(**payload['manifest']['architecture']).cuda().eval()
    model.load_state_dict(payload['model'], strict=True)
    model.requires_grad_(False)
    return model, payload['manifest'], commit


class FrozenVision:
    def __init__(self, source):
        dependencies()
        from stripsnav.io import load_committed
        from nesy_vla.runtime import load_checkpoint
        from nesy_vla.model import NavigationModel
        location = Path(source['encoder_source'])
        self.cfg = read(location.parent / 'config.json')
        assert read(location / 'COMMITTED') == source['encoder_source_commit']
        visual = NavigationModel(self.cfg).eval()
        load_checkpoint(location, visual)
        vision = visual.grounder.vision
        if hasattr(vision, 'backbone'):
            vision = vision.backbone
        assert isinstance(vision, torch.nn.Sequential) and len(vision) == 8
        component = Path(source['component_checkpoint'])
        assert read(component / 'COMMITTED') == source['component_commit']
        payload = load_committed(component)
        if payload['visual_tail'] is not None:
            vision[6:].load_state_dict(payload['visual_tail'])
        self.vision = vision.cuda().eval().requires_grad_(False)
        assert sha(source['vocabulary_path']) == source['vocabulary_sha256']
        self.vocabulary = torch.load(source['vocabulary_path'], map_location='cuda', weights_only=True)
        self.calls = 0
        self.frames = 0

    @torch.no_grad()
    def encode(self, images):
        from nesy_vla.runtime import image_tensor
        self.calls += 1
        self.frames += len(images)
        grids = []
        # Keep original single-frame vocabulary assignment numerics.
        for image in images:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                features = F.adaptive_avg_pool2d(self.vision(image_tensor(image, self.cfg['image_size'], 'cuda')), (7, 7)).flatten(2).transpose(1, 2)
            v = self.vocabulary
            z = F.normalize(((features.float() - v['mean']) / v['sd']) @ v['axes'], dim=-1)
            grids.append(torch.cdist(z.flatten(0, 1), v['centers']).argmin(1))
        return torch.stack(grids)


def make_bank():
    cfg = dependencies()
    from nesy_vla.runtime import DataBank
    return DataBank(cfg)
