from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
model=(BASE/'onpolicy_refresh_v60/model.py').read_text(encoding='utf-8')
model+='\n\n'+(HERE/'relational_rule.py').read_text(encoding='utf-8')
model+='\n\n'+(HERE/'world_extension.py').read_text(encoding='utf-8')
(HERE/'model.py').write_text(model,encoding='utf-8')
for name in ('planner.py','program_archive.py','evaluate.py'):
    (HERE/name).write_bytes((BASE/'onpolicy_refresh_v60'/name).read_bytes())
print('Built candidate with unchanged expected categorical planner and full80 evaluator')
