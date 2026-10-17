import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import torch
from PIL import Image
from nesy_vla.generator import ProgramDecoder
from nesy_vla.logic import SoftLogic
from nesy_vla.policy import EpisodePolicy
from nesy_vla.runtime import public_feedback


class FakeModel:
    def __init__(self):
        self.cfg={'revision_interval':10};self.device='cpu';self.head=ProgramDecoder(hidden=32)
        self.logic=SoftLogic();self.calls=[]
    def generator(self,task,images,feedback,previous):
        self.calls.append(len(feedback))
        return self.head(torch.zeros(1,4,32),previous)
    def step(self,p,image,memory,prior):
        result=self.logic.infer(p,torch.full((16,),.5),torch.ones(16))
        result['value']=torch.tensor(0.);result['observation']=torch.full((16,),.5);return result
    def advance(self,p,result,a,feedback,memory):
        return (torch.full((16,),.5),torch.ones(16)),memory,torch.tensor(0.)
    def transition_target(self,p,image,memory,tokens=None):return torch.full((16,),.5),torch.ones(16)


class ScheduleTests(unittest.TestCase):
    def test_exact_executed_action_boundaries(self):
        model=FakeModel();policy=EpisodePolicy(model,'chair');image=Image.new('RGB',(32,32))
        with torch.no_grad():
            for _ in range(32):
                action,result,row=policy.choose(image,action=4)
                policy.observe(row,result,image,{'feedback':{},'metrics':{}})
        self.assertEqual(model.calls,[0,10,20,30])
        self.assertEqual([len(c['rows']) for c in policy.chunks],[10,10,10,2])
        self.assertEqual([c['program'].version for c in policy.chunks],[0,1,2,3])

    def test_private_geometry_never_enters_feedback(self):
        image=Image.new('RGB',(32,32))
        public,_=public_feedback({'feedback':{'collision':True,'distance':.1,'goal_position':[0,0]},
            'metrics':{'eligible':True},'reward':100},image,image)
        self.assertEqual(set(public),{'collision','visual_change'})


if __name__=='__main__':unittest.main()
