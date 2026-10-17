from pathlib import Path
here=Path(__file__).parent;parent=here.parent
for name in ('model.py','planner.py','program_archive.py','evaluate.py','data.py','objectives.py','train.py','storage.py','supervisor.py'):
    (here/name).write_text((parent/'goal_aware_symbols_v52'/name).read_text(encoding='utf-8'),encoding='utf-8')
(here/'onpolicy_worker.py').write_text((parent/'probabilistic_rules_v40/onpolicy_worker.py').read_text(encoding='utf-8'),encoding='utf-8')
s=(parent/'probabilistic_rules_v40/collect_onpolicy.py').read_text(encoding='utf-8')
s=s.replace('from model import FactoredWorld, Config','from model import FactoredWorld, Config, hard')
s=s.replace('self.program = None','self.program = None\n        self.context = None\n        self.grounding_allowed = False')
s=s.replace("observed = int(self.model.observe(raw).argmax(-1))\n            self.features.append(raw[0])\n            self.observations.append(observed)","observed_global = int(self.model.observe(raw).argmax(-1))\n            observed = int(self.model.observe(raw,self.context).argmax(-1))\n            self.features.append(raw[0])\n            self.observations.append(observed_global)")
old='''                memory_logits, _, _ = self.model.memory_sequence(observations, previous)
                self.memory = int(memory_logits[0, -1].argmax(-1))
                context = self.model.context_sequence(history, observations, previous, self.task)[:, -1]'''
new='''                context = self.model.context_sequence(history, observations, previous, self.task)[:, -1]
                self.context = context
                local_logits,evidence,_ = self.model.observation_encoding(history,context)
                observations = hard(local_logits)
                observed = int(local_logits[0,-1].argmax(-1))
                self.grounding_allowed = True
                try:
                    memory_logits, _, _ = self.model.memory_sequence(observations, previous,evidence)
                finally:
                    self.grounding_allowed = False
                self.memory = int(memory_logits[0, -1].argmax(-1))'''
assert old in s;s=s.replace(old,new)
s=s.replace("parser.add_argument('--workers', type=int, default=4)","parser.add_argument('--workers', type=int, default=4)\n    parser.add_argument('--rank',type=int,required=True)")
s=s.replace("assert os.environ['CUDA_VISIBLE_DEVICES'] == '6' and 1 <= args.workers <= 4", "assert args.rank in (0,1) and os.environ['CUDA_VISIBLE_DEVICES'] == str(6+args.rank) and 1 <= args.workers <= 4")
s=s.replace("out.name == 'onpolicy_successors_v40'","out.name == 'onpolicy_successors_v54'")
s=s.replace("(out / 'collector.lock')","(out / f'collector_rank{args.rank}.lock')")
s=s.replace('selected_tasks(bank, args.tasks, 401017)','selected_tasks(bank, args.tasks, 541017)')
s=s.replace("policy='v40 exact probabilistic discrete derivation'", "policy='v52 best2000 stable exact probability derivation with task-conditioned symbols'")
s=s.replace('seed=401017','seed=541017,world_size=2')
s=s.replace("atomic(out/'manifest.json', provenance)","if args.rank==0:atomic(out/'manifest.json', provenance)\n    tasks=tasks[args.rank::2]")
s=s.replace("folder=out/f'worker_{i}'", "folder=out/f'worker_{args.rank}_{i}'")
s=s.replace("out/'status.json'", "out/f'status_rank{args.rank}.json'")
s=s.replace("handle = model.inverse.register_forward_pre_hook(reject_inverse)", """handle = model.inverse.register_forward_pre_hook(reject_inverse)
    active_policy=[None]
    def reject_memory_grounding(*_):
        assert active_policy[0] is not None and active_policy[0].grounding_allowed, 'Memory grounded outside renewal'
    memory_handle=model.memory.register_forward_pre_hook(reject_memory_grounding)""")
s=s.replace("action,raw=policies[i].choose(rgb[i])", "active_policy[0]=policies[i]\n                        action,raw=policies[i].choose(rgb[i])\n                        active_policy[0]=None")
s=s.replace("handle.remove();gate.close()", "memory_handle.remove();handle.remove();gate.close()")
(here/'collect_onpolicy.py').write_text(s,encoding='utf-8')
print('Prepared v52-compatible two-card collection. No execution or training launched.')
