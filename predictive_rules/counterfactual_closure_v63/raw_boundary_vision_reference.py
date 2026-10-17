"""One canonical CNN pass supplies patch IDs and pre-quantization evidence."""
import torch
from torch.nn import functional as F
from .fast_vision import BoundaryGraphEncoder

class RawBoundaryGraphEncoder(BoundaryGraphEncoder):
    def _encode_tensor(self):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            features=F.adaptive_avg_pool2d(self.frontend.vision(self.input),(7,7)).flatten(2).transpose(1,2)
        v=self.frontend.vocabulary;z=F.normalize(((features.float()-v['mean'])/v['sd'])@v['axes'],dim=-1)
        self.feature_output=z[0].to(torch.float16)
        return torch.cdist(z.flatten(0,1),v['centers']).argmin(1)

    @torch.no_grad()
    def encode_joint(self,images):
        codes=[];features=[]
        for image in images:
            codes.append(super().encode([image])[0]);features.append(self.feature_output.clone())
        return torch.stack(codes),torch.stack(features)
