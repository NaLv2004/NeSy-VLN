from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'goal_equivalence_v57'
s=(BASE/'model.py').read_text(encoding='utf-8')+'\n'+(HERE/'ensemble_extension.py').read_text(encoding='utf-8')
(HERE/'model.py').write_text(s,encoding='utf-8')
(HERE/'planner.py').write_text((HERE/'robust_planner.py').read_text(encoding='utf-8'),encoding='utf-8')
(HERE/'program_archive.py').write_text((BASE/'program_archive.py').read_text(encoding='utf-8'),encoding='utf-8')
s=(BASE/'evaluate.py').read_text(encoding='utf-8')
old="(program.outcome_probability[:,0,go]>0)"
new="(program.outcome_probability[:,:,0,go].min(0)>0)"
assert old in s;s=s.replace(old,new)
s=s.replace("'Probabilistic discrete-rule derivation with all outcomes and a separately recorded successful witness'","'Rectangular robust probability derivation over three exported rule tables, with adversarial-row witness'")
(HERE/'evaluate.py').write_text(s,encoding='utf-8')
print('Built full80 evaluation candidate; no evaluation or formal training started.')
