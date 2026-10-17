"""Create a separate implementation; never edits the live v54 source."""
from pathlib import Path
import ast
import shutil

SRC=Path(__file__).resolve().parent
DST=SRC.parent/'temporal_symbols_v55';DST.mkdir(exist_ok=True)
for name in ('model.py','planner.py','program_archive.py','evaluate.py','data.py','objectives.py','train.py','supervisor.py','storage.py'):
    shutil.copy2(SRC/name,DST/name)

p=DST/'model.py';s=p.read_text(encoding='utf-8')
s=s.replace('        self.detach_copy_evidence=True', '''        # Training-only discriminator on hard symbolic premises. It never
        # receives images/context/action IDs and is forbidden at runtime.
        self.temporal_contrast=nn.Sequential(nn.Linear(c.memories+2*c.observations,128),
            nn.Tanh(),nn.Linear(128,1))
        self.detach_copy_evidence=True''')
s=s.replace('    def load_v48(self, state):', '''    def load_v54(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        assert not unexpected and set(missing)=={
            'temporal_contrast.0.weight','temporal_contrast.0.bias',
            'temporal_contrast.2.weight','temporal_contrast.2.bias'}, (missing,unexpected)

    def load_v48(self, state):''')
p.write_text(s,encoding='utf-8')

p=DST/'data.py';s=p.read_text(encoding='utf-8')
s=s.replace('        bt=[];bs=[];tasks=[]','''        bt=[];bs=[];tasks=[]
        negatives=np.zeros_like(successors)''')
s=s.replace('            successors[b]=d[\'successor_features\'][t]', '''            successors[b]=d['successor_features'][t]
            # Marginal views from this same recorded episode. These are
            # training-only examples, not "impossible" or collision labels.
            valid_positions=np.flatnonzero(d['probe_valid']).tolist()
            assert valid_positions
            for a in range(17):
                nt=rng.choice(valid_positions);na=rng.randrange(17)
                negatives[b,a]=d['successor_features'][nt,na]''')
s=s.replace('branch_t=np.asarray(bt),branch_start=np.asarray(bs)', 'branch_negative_features=negatives,branch_t=np.asarray(bt),branch_start=np.asarray(bs)')
p.write_text(s,encoding='utf-8')

p=DST/'objectives.py';s=p.read_text(encoding='utf-8')
s=s.replace('    total=hybrid_replay_loss+', '''    # Density-ratio objective from Markov state abstraction learning. Both
    # real and marginal future images use the CURRENT program's definitions.
    from temporal_objective import temporal_ratio
    ratio,ratio_metrics=temporal_ratio(model,m0,inverse_current,so,successors,
        data['branch_negative_features'],bc)
    total=ratio+hybrid_replay_loss+''')
s=s.replace('    return total,metrics','    metrics.update(ratio_metrics)\n    return total,metrics')
p.write_text(s,encoding='utf-8')

p=DST/'evaluate.py';s=p.read_text(encoding='utf-8')
s=s.replace('    inverse_handle=model.inverse.register_forward_pre_hook(reject_inverse)', '''    inverse_handle=model.inverse.register_forward_pre_hook(reject_inverse)
    contrast_handle=model.temporal_contrast.register_forward_pre_hook(reject_inverse)''')
s=s.replace('        inverse_handle.remove();grounding_handle.remove();gate.close()', '        contrast_handle.remove();inverse_handle.remove();grounding_handle.remove();gate.close()')
p.write_text(s,encoding='utf-8')

for name in ('train.py','supervisor.py','storage.py'):
    p=DST/name;s=p.read_text(encoding='utf-8')
    s=s.replace('src_hybrid_replay_v54','src_temporal_symbols_v55').replace('run_hybrid_replay_v54','run_temporal_symbols_v55')
    s=s.replace("architecture='hybrid_replay_v54'", "architecture='temporal_symbols_v55'")
    s=s.replace("in ('hybrid_replay_v54',)","in ('temporal_symbols_v55',)")
    s=s.replace('29754','29755').replace('391017','551017')
    if name=='train.py':
        s=s.replace("model.load_state_dict(previous['model'],strict=True)","model.load_v54(previous['model'])")
        s=s.replace("optimizer_initialization='v52 best2000 exact weights; fresh AdamW; same architecture and rates'", "optimizer_initialization='Pinned v54 best weights plus new training-only temporal discriminator; fresh AdamW; unchanged representation/rule rates'")
        s=s.replace("source_model='v52 best2000; current-policy factual branch aggregation plus runtime-matched predicted-memory/observed-state supervision'", "source_model='v54 best; add density-ratio training of hard symbolic states; all runtime rule/planner behavior preserved before optimization'")
        s=s.replace("    first=1", "    manifest.update(temporal_density_ratio_weight=1.0,temporal_classifier_runtime_calls=0,temporal_negatives='Half same-episode marginal images, half cross-batch images; all re-encoded under current fixed definitions',temporal_reference='https://github.com/camall3n/markov-state-abstractions',temporal_reference_commit='9495ad2880a89fa9dcb99a870f301f5b379bda5c',no_oracle_distance_loss=True)\n    first=1")
    if name=='supervisor.py':
        s=s.replace("str(ROOT/'onpolicy_successors_v54/source.pt')", "str(ROOT/'diagnostic_temporal_symbols_v55/source.pt')")
    p.write_text(s,encoding='utf-8')
for p in DST.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
print(DST)
