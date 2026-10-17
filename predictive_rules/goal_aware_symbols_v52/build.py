"""One-time isolated adaptation; never modify a running source tree."""
from pathlib import Path
here=Path(__file__).parent
for name in ('model.py','objectives.py','train.py','evaluate.py','data.py','storage.py','program_archive.py','supervisor.py','inspect_run.py'):
    path=here/name
    text=path.read_text(encoding='utf-8')
    text=text.replace('contextual_symbols_v48','goal_aware_symbols_v52')
    if name=='model.py':
        old='        self.task = nn.Embedding(2,c.context)'
        assert old in text
        text=text.replace(old,old+'\n        self.abstraction_task = nn.Embedding(2,c.context)\n        nn.init.zeros_(self.abstraction_task.weight)')
        old='''        # Observational dynamics cannot change merely because a different task
        # was requested. Task conditioning is reserved for the generated goal.
        return self.context_history(x.float())[0]'''
        new='''        # The physical scene does not change with a task. The LEARNED symbolic
        # basis may focus on different evidence for different tasks. All current
        # and future labels and every generated rule use this SAME fixed basis.
        return self.context_history(x.float())[0]+self.abstraction_task(task)[:,None]'''
        assert old in text;text=text.replace(old,new)
        anchor='    def export_config(self):'
        method='''    def load_v48(self, state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        assert not unexpected and missing==['abstraction_task.weight'], (missing,unexpected)
        nn.init.zeros_(self.abstraction_task.weight)

'''
        assert anchor in text;text=text.replace(anchor,method+anchor)
    if name=='train.py':
        text=text.replace("model.load_v43(previous['model'])","model.load_v48(previous['model'])")
        text=text.replace("dynamics_task_invariant=True,task_conditioning='Only the goal generator sees task identity; world dynamics and learned state definitions cannot depend on the requested goal'",
            "dynamics_task_invariant=False,task_conditioning='Task-dependent anonymous symbolic basis and matching generated dynamics; no task-specific physical effects, labels recomputed in the same basis'")
        text=text.replace("v43 best weights initialize existing modules; new full definition vectors zero-initialized; fresh AdamW",
            "v48 best500 weights initialize existing modules; new task-conditioned abstraction embedding zero-initialized; fresh AdamW")
        text=text.replace("source_model='v43 best3500; no Qwen vision substitution'","source_model='v48 best500; no Qwen vision substitution'")
        text=text.replace("runtime_action_selection='Unchanged NumPy exhaustive probability derivation from exported integer table'",
            "runtime_action_selection='NumPy exhaustive probability derivation with normalized float64 dual success/failure arithmetic v50'")
        text=text.replace("references=['https://github.com/midi-lab/acdf','https://arxiv.org/abs/2107.00110']",
            "references=['https://github.com/midi-lab/acdf','https://proceedings.mlr.press/v119/nair20a.html']")
    if name=='objectives.py':
        text=text.replace("metrics.update(definition_delta_rms=delta.detach().square().mean().sqrt(),",
            "metrics.update(abstraction_task_rms=model.abstraction_task.weight.detach().square().mean().sqrt(),\n        task_basis_separation=(model.abstraction_task.weight[0]-model.abstraction_task.weight[1]).detach().norm(),\n        definition_delta_rms=delta.detach().square().mean().sqrt(),")
    if name=='supervisor.py':
        text=text.replace('29748','29752').replace("diagnostic_sparse_support_v44/source.pt","diagnostic_goal_aware_symbols_v52/source.pt")
    path.write_text(text,encoding='utf-8')
print('Prepared isolated goal-conditioned anonymous-state trainer with stable symbolic planning.')
