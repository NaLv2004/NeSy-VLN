import json,tempfile,unittest
from pathlib import Path
import numpy as np
from predictive_nesy.interpreter import Program,export_program
from predictive_nesy.belief_interpreter import joint,successors,BeliefController,plan
from test_contract import toy_program


class BeliefTests(unittest.TestCase):
    def test_same_parent_is_one_variable_not_two_independent_draws(self):
        p=toy_program()
        p.parents[:]=[[0,0],[1,1]]
        state=np.array([[.6,.4,0,0],[.2,.8,0,0]],np.float32)
        j=joint(p,state)
        np.testing.assert_allclose(j[0],np.diag(state[0]),atol=1e-7)
        np.testing.assert_allclose(j[1],np.diag(state[1]),atol=1e-7)
        self.assertEqual(float(j[0,0,1]),0)

    def test_uncertainty_propagates_even_when_argmax_is_unchanged(self):
        p=toy_program();p.parents[:]=[[0,0],[1,1]]
        for f in range(2):
            p.transitions[f,1,0,0]=[.8,.2,0,0]
            p.transitions[f,1,1,1]=[0,1,0,0]
        initial=np.eye(4,dtype=np.float32)[[0,0]]
        first=successors(p,initial)[1]
        second=successors(p,first)[1]
        np.testing.assert_array_equal(first.argmax(-1),second.argmax(-1))
        np.testing.assert_allclose(second[:,1],[.36,.36],atol=1e-6)
        self.assertFalse(np.allclose(first,second))

    def test_exported_belief_trajectory_independent_audit(self):
        from audit_trace import audit
        p=toy_program();c=BeliefController();r=c.revise(p,[0,0])
        self.assertEqual(r['actions'],[2,1,0])
        steps=[c.next_action()[1] for _ in range(3)]
        trace=dict(episode={'episode_id':'belief_contract'},steps=steps,neural_calls_outside_boundaries=0,
            programs=[dict(start=0,observed_frame_indices=[0],file='program.npz',initial_state=[0,0],plan=r)])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p.save(root/'program.npz');(root/'trace.json').write_text(json.dumps(trace))
            self.assertEqual(audit(root)['steps'],3)
        with self.assertRaises(TypeError):c.next_action(np.zeros((224,224,3)))


class MemoryModelTests(unittest.TestCase):
    def test_torch_numpy_probabilistic_interpreters_match(self):
        import torch
        from predictive_nesy.model_memory import MemoryConfig,PredictiveMemoryModel,transition_all
        model=PredictiveMemoryModel(MemoryConfig(states=4,actions=3,hidden=32))
        codes=torch.randint(0,64,(2,1,49));prev=torch.full((2,1),3)
        program=model.generate(torch.tensor([0,1]),codes,prev,torch.ones(2,dtype=torch.long),start=0)
        state=torch.rand(2,8,4);state=state/state.sum(-1,keepdim=True)
        expected=transition_all(program,state)
        for i in range(2):
            p=export_program(program,i)
            actual=successors(p,state[i].numpy())
            np.testing.assert_allclose(actual,expected[i].detach().numpy(),atol=1e-6)
            np.testing.assert_allclose(actual.sum(-1),1,atol=1e-6)

    def test_memory_target_causal_and_gradient_flows(self):
        import torch
        from predictive_nesy.model_memory import MemoryConfig,PredictiveMemoryModel
        from predictive_nesy.training_memory import objective
        device='cuda' if torch.cuda.is_available() else 'cpu'
        m=PredictiveMemoryModel(MemoryConfig(states=4,hidden=32)).to(device)
        b,t=2,11
        batch=dict(start=torch.zeros(b,dtype=torch.long,device=device),task=torch.zeros(b,dtype=torch.long,device=device),
            history=torch.randint(0,64,(b,1,49),device=device),previous_actions=torch.full((b,1),17,device=device),
            history_length=torch.ones(b,dtype=torch.long,device=device),targets=torch.randint(0,64,(b,t,49),device=device),
            actions=torch.randint(1,17,(b,t),device=device),valid=torch.ones(b,t,dtype=torch.bool,device=device),
            supervised=torch.ones(b,t,dtype=torch.bool,device=device),teacher_ids=torch.randint(0,4,(b,t,4),device=device),
            goal=torch.zeros(b,t,dtype=torch.bool,device=device),value=torch.ones(b,t,device=device),
            risks=torch.full((b,t,17),-1.,device=device))
        batch['targets'][:,0]=batch['history'][:,0]
        with torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda'):
            loss,metrics=objective(m,batch)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.memory_features.parameters() if p.grad is not None),0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.transitions.parameters() if p.grad is not None),0)
        program=m.generate(batch['task'],batch['history'],batch['previous_actions'],batch['history_length'],start=0)
        before=m.training_targets(program,batch['targets'],batch['actions'])
        changed=batch['targets'].clone();changed[:,5:]=torch.randint(0,64,(b,6,49),device=device)
        after=m.training_targets(program,changed,batch['actions'])
        self.assertTrue(torch.equal(before[:,:5],after[:,:5]))
        self.assertTrue(torch.equal(before[:,0],m.ground(program,batch['targets'][:,0])))


if __name__=='__main__':unittest.main()
