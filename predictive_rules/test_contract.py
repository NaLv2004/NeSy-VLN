"""Contract tests, including an independent planner reproduction and leakage traps."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np

from predictive_nesy.interpreter import Program, BlockController, plan, transition


def toy_program():
    # Arbitrary state IDs encode a small graph. The rewarding path is [2,1,0].
    # Action 1 is a tempting dead end at the root; future-state planning is needed.
    f,a,k=2,3,4
    table=np.zeros((f,a,k,k,k),dtype=np.float32)
    for factor in range(f):
        for action in range(a):
            for u in range(k):
                for v in range(k):
                    successor=u
                    if action==1:
                        successor={0:3,1:2,2:2,3:3}[u]
                    elif action==2:
                        successor={0:1,1:1,2:2,3:3}[u]
                    table[factor,action,u,v,successor]=1
    goal=np.zeros((1,k,k),dtype=np.float32);goal[0,2,2]=1
    value=np.zeros_like(goal);value[0,0,0]=.8;value[0,1,1]=.9;value[0,2,2]=1
    return Program(np.array([[0,1],[1,0]]),table,np.zeros((f,a,k,k)),goal,value,np.ones(1),
                   np.zeros((f,k,2)),0).validate()


class PureInterpreterTests(unittest.TestCase):
    def test_independent_audit_rejects_injected_state_correction(self):
        from audit_trace import audit
        controller=BlockController()
        generated=toy_program()
        result=controller.revise(generated,[0,0])
        steps=[controller.next_action()[1] for _ in range(3)]
        trace=dict(episode={'episode_id':'contract_fixture'},programs=[dict(start=0,
            observed_frame_indices=[0],file='program.npz',initial_state=[0,0],plan=result)],
            steps=steps,neural_calls_outside_boundaries=0)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);generated.save(root/'program.npz')
            (root/'trace.json').write_text(json.dumps(trace))
            self.assertEqual(audit(root)['steps'],3)
            trace['steps'][1]['after']=[0,0]
            (root/'trace.json').write_text(json.dumps(trace))
            with self.assertRaises(AssertionError):audit(root)

    def test_multistep_plan_and_export_replay(self):
        p=toy_program()
        result=plan(p,[0,0],horizon=10,beam_width=16)
        self.assertEqual(result['actions'],[2,1,0])
        self.assertEqual(result['states'],[[0,0],[1,1],[2,2],[2,2]])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'program.npz';p.save(path)
            restored=Program.load(path)
            self.assertEqual(plan(restored,[0,0],10,16),result)

    def test_state_correction_only_at_boundary(self):
        controller=BlockController()
        controller.revise(toy_program(),[0,0])
        self.assertEqual(controller.next_action()[0],2)
        with self.assertRaises(ValueError):
            controller.revise(toy_program(),[2,2])
        self.assertEqual(controller.next_action()[0],1)
        self.assertEqual(controller.next_action()[0],0)
        with self.assertRaises(ValueError):
            controller.next_action()

    def test_intermediate_observations_cannot_enter_controller(self):
        c1,c2=BlockController(),BlockController()
        c1.revise(toy_program(),[0,0]);c2.revise(toy_program(),[0,0])
        for _ in range(3):
            random_external_frame=np.random.randint(0,256,(224,224,3),dtype=np.uint8)
            with self.assertRaises(TypeError):
                c1.next_action(random_external_frame)
            self.assertEqual(c1.next_action(),c2.next_action())


class NeuralContractTests(unittest.TestCase):
    def test_ten_step_supervision_backward_and_future_input_isolation(self):
        import torch
        from predictive_nesy.model import ModelConfig,GeneratedWorldModel
        from predictive_nesy.training import objective
        device='cuda' if torch.cuda.is_available() else 'cpu'
        model=GeneratedWorldModel(ModelConfig(states=4,hidden=32)).to(device)
        b,t=2,11
        batch=dict(start=torch.zeros(b,dtype=torch.long,device=device),task=torch.zeros(b,dtype=torch.long,device=device),
            history=torch.randint(0,64,(b,1,49),device=device),previous_actions=torch.full((b,1),17,device=device),
            history_length=torch.ones(b,dtype=torch.long,device=device),targets=torch.randint(0,64,(b,t,49),device=device),
            actions=torch.randint(1,17,(b,t),device=device),valid=torch.ones(b,t,dtype=torch.bool,device=device),
            supervised=torch.ones(b,t,dtype=torch.bool,device=device),teacher_ids=torch.randint(0,4,(b,t,4),device=device),
            goal=torch.zeros(b,t,dtype=torch.bool,device=device),value=torch.ones(b,t,device=device),
            risks=torch.full((b,t,17),-1.,device=device))
        batch['targets'][:,0]=batch['history'][:,0]
        captured=[]
        original=model.generate
        def capture(*args,**kwargs):
            result=original(*args,**kwargs)
            captured.append(result['transitions'].detach().clone())
            return result
        model.generate=capture
        with torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda'):
            loss,metrics=objective(model,batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.transitions.parameters() if p.grad is not None),0)
        batch['targets'][:,1:]=torch.randint(0,64,(b,t-1,49),device=device)
        with torch.no_grad(),torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda'):
            objective(model,batch)
        self.assertTrue(torch.equal(captured[0],captured[1]))

    def test_generated_table_consistency_gradients_and_boundary_enforcement(self):
        import torch
        from predictive_nesy.model import ModelConfig, GeneratedWorldModel, transition_all, straight_through
        from predictive_nesy.interpreter import export_program
        from predictive_nesy.gate import NeuralBoundaryGate
        torch.manual_seed(15)
        model=GeneratedWorldModel(ModelConfig(states=4,actions=3,hidden=32)).eval()
        gate=NeuralBoundaryGate([model])
        history=torch.randint(0,64,(2,1,49));tasks=torch.tensor([0,1]);prev=torch.full((2,1),3)
        with gate.boundary(0):
            generated=model.generate(tasks,history,prev,torch.ones(2,dtype=torch.long),start=0)
            state=straight_through(model.ground(generated,history[:,0]))
        predictions=transition_all(generated,state)
        for b in range(2):
            exported=export_program(generated,b)
            for a in range(3):
                ids,probs=transition(exported,state[b].argmax(-1).numpy(),np.asarray(a))
                np.testing.assert_allclose(probs,predictions[b,a].detach().numpy(),atol=1e-6)
        loss=-predictions[...,0].clamp_min(1e-8).log().mean()
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.transitions.parameters() if p.grad is not None),0)
        with self.assertRaises(RuntimeError):
            model.ground(generated,history[:,0])
        with self.assertRaises(RuntimeError):
            with gate.boundary(1):pass
        gate.close()
        self.assertFalse(any('basis' in n or 'conclusion' in n for n,_ in model.named_parameters()))

    def test_future_frames_do_not_change_generated_program(self):
        import torch
        from predictive_nesy.model import ModelConfig,GeneratedWorldModel
        model=GeneratedWorldModel(ModelConfig(states=4,actions=3,hidden=32)).eval()
        history=torch.randint(0,64,(1,11,49));actions=torch.zeros(1,11,dtype=torch.long)
        inputs=dict(tasks=torch.tensor([0]),history=history,previous_actions=actions,lengths=torch.tensor([11]),start=10)
        first=model.generate(**inputs)
        # Future records exist in the training batch, but this API has no slot
        # for them and generated tensors must remain identical without them.
        future=torch.randint(0,64,(1,11,49));future.random_(0,64)
        second=model.generate(**inputs)
        for key in ['definitions','transitions','parents','goals','values','risks','weights']:
            self.assertTrue(torch.equal(first[key],second[key]),key)
        with self.assertRaises(ValueError):model.generate(**dict(inputs,start=11))


if __name__=='__main__':
    unittest.main()
