from pathlib import Path
HERE=Path(__file__).resolve().parent
source=(HERE.parent/'onpolicy_refresh_v60/collect_onpolicy.py').read_text(encoding='utf-8')
prefix=source[:source.index('@torch.no_grad()\ndef main():')]
prefix=prefix.replace('import argparse','import argparse\nimport copy')
prefix=prefix.replace('onpolicy_worker.py','closure_worker.py')
prefix=prefix.replace('def choose(self, rgb):','def choose(self, rgb, defer_planning=False):')
marker="        proof = plan(self.program, self.memory, observed, horizon=32-step)"
assert marker in prefix
prefix=prefix.replace(marker,"""        prepared = (observed, renew, probability)
        raw = raw[0].cpu().numpy()
        if defer_planning:
            return prepared, raw
        return self.complete_choice(prepared, raw)

    def complete_choice(self, prepared, raw):
        # Pure NumPy rule-table planning; independent environments may run
        # this method concurrently. It never touches a neural module or CUDA.
        observed, renew, probability = prepared
        step = len(self.actions)
        proof = plan(self.program, self.memory, observed, horizon=32-step)""")
prefix=prefix.replace("return None, raw[0].cpu().numpy()","return None, raw")
prefix=prefix.replace("return action, raw[0].cpu().numpy()","return action, raw")
(HERE/'collect_closure.py').write_text(prefix+(HERE/'collector_body.txt').read_text(encoding='utf-8'),encoding='utf-8')
for name in ('model.py','planner.py','data.py','evaluate.py','program_archive.py'):
    (HERE/name).write_bytes((HERE.parent/'onpolicy_refresh_v60'/name).read_bytes())
print('Built fixed-source two-action observation collector')
