import unittest,inspect,itertools,tempfile
from pathlib import Path
import numpy as np
import torch
from predictive_nesy.graph_subgoal_model import GraphSubgoalModel,Config,straight_through
from predictive_nesy.graph_subgoal_training import objective,transition,reachability,proof_log_support
from predictive_nesy import graph_subgoal_interpreter as logic
from predictive_nesy.gate import NeuralBoundaryGate


class GraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(76);self.model=GraphSubgoalModel(Config(states=8,width=64,hidden=32,decoder_hidden=64))
        self.data=dict(task=torch.tensor([0,1]),start=torch.zeros(2,dtype=torch.long),history=torch.randint(0,64,(2,1,49)),
            previous_actions=torch.full((2,1),17),history_length=torch.ones(2,dtype=torch.long),targets=torch.randint(0,64,(2,11,49)),
            actions=torch.tensor([[1,2,0,0,0,0,0,0,0,0]]*2),action_valid=torch.tensor([[1,1,1,0,0,0,0,0,0,0]]*2).bool(),
            state_valid=torch.tensor([[1,1,1,1,0,0,0,0,0,0,0]]*2).bool(),supervised=torch.ones(2,10).bool(),goal_index=torch.tensor([3,3]))

    def program(self):
        d=self.data;return self.model.generate(d['task'],d['history'],d['previous_actions'],d['history_length'],start=0)

    def test_only_generated_rules_definitions_and_goal(self):
        p=self.program();export=logic.export_program(p)
        self.assertEqual(set(export.__dict__),set(logic.FIELDS)|{'start','interval'})
        self.assertEqual(export.subgoal.tolist(),p['subgoal_logits'][0].argmax(-1).tolist())
        self.assertFalse(any(x in n for n in self.model.state_dict() for x in ['actor','reward','value','collision']))
        with tempfile.TemporaryDirectory() as path:
            export.save(Path(path)/'rules.npz');self.assertEqual(export.digest(),logic.Program.load(Path(path)/'rules.npz').digest())

    def test_runtime_matches_training_transitions(self):
        p=self.program();state=straight_through(self.model.ground(p,self.data['targets'][:,0]).squeeze(1))
        for a in range(17):
            target=transition(p,state,torch.full((2,),a)).argmax(-1)
            for i in range(2):
                actual=logic.successors(logic.export_program(p,i),np.array([state[i].argmax().item()]))[0][a,0]
                self.assertEqual(actual,target[i].item())

    def test_proof_supervision_reaches_rule_and_goal_generators(self):
        loss,metrics=objective(self.model,self.data);loss.backward();self.assertTrue(torch.isfinite(loss))
        for m in [self.model.subgoal,self.model.transitions,self.model.definitions,self.model.observation]:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None),0)
        self.assertIn('proof_action_imitation',metrics)

    def test_neural_boundaries(self):
        gate=NeuralBoundaryGate([self.model])
        try:
            with gate.boundary(0):p=self.program();export=logic.export_program(p)
            with self.assertRaises(RuntimeError):self.program()
            logic.plan(export,np.array([0]))
            self.assertEqual(list(inspect.signature(logic.Controller().next_action).parameters),[])
        finally:gate.close()

    @staticmethod
    def toy():
        t=np.zeros((4,3,4),np.float32)
        for s in range(4):t[s,:,s]=1
        for s,a,n in [(0,2,1),(1,1,2),(2,0,3)]:t[s,a]=0;t[s,a,n]=1
        return logic.Program(t,np.zeros((4,8)),np.array([3]),0)

    def test_complete_goal_derivation(self):
        p=self.toy();controller=logic.Controller();proof=controller.revise(p,np.array([0]))
        self.assertEqual(proof['actions'],[2,1,0])
        for a in proof['actions']:self.assertEqual(controller.next_action()[0],a)
        self.assertTrue(controller.finished);self.assertEqual(controller.current.tolist(),[3])

    def test_unreachable_has_no_fallback(self):
        p=self.toy();p.transitions[2,0]=[0,0,1,0]
        proof=logic.plan(p,np.array([0]));self.assertFalse(proof['found']);self.assertEqual(proof['actions'],[])

    def test_anonymous_state_permutation(self):
        p=self.toy();permutation=np.array([2,3,1,0]);t=np.zeros_like(p.transitions)
        for s,a,n in itertools.product(range(4),range(3),range(4)):t[permutation[s],a,permutation[n]]=p.transitions[s,a,n]
        perm=logic.Program(t,p.definitions,np.array([permutation[3]]),0)
        self.assertEqual(logic.plan(p,np.array([0]))['actions'],logic.plan(perm,np.array([permutation[0]]))['actions'])

    def test_soft_logic_reachability_equals_hard_exhaustive_boolean(self):
        p=self.toy();table=torch.tensor(p.transitions)[None];goal=torch.nn.functional.one_hot(torch.tensor([3]),4).float()
        levels=reachability({'transitions':table},goal,4)
        for h in range(1,5):
            for s in range(4):
                found=logic.plan(p,np.array([s]),horizon=h)['found']
                self.assertEqual(bool(levels[h][0,s].max()),found)

    def test_training_and_execution_choose_same_goal_proof(self):
        rng=np.random.default_rng(113)
        for trial in range(8):
            table=rng.random((6,3,6)).astype(np.float32);table/=table.sum(-1,keepdims=True)
            program=logic.Program(table,np.zeros((6,8)),np.array([trial%6]),0)
            goal=torch.nn.functional.one_hot(torch.tensor([trial%6]),6).float()
            for horizon in range(1,6):
                derived=proof_log_support({'transitions':torch.tensor(table)[None]},goal,horizon)[horizon]
                for state in range(6):
                    route=logic.plan(program,np.array([state]),horizon=horizon)
                    if route['found']:self.assertEqual(route['actions'][0],derived[0,state].argmax().item())


if __name__=='__main__':unittest.main()
