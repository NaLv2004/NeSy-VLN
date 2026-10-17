import ast,shutil
from pathlib import Path
BASE=Path(__file__).parent;dst=BASE/'full_history_v79';dst.mkdir(exist_ok=True)
for p in (BASE/'expanded_history_v76').glob('*.py'):
    if p.name not in ('repair_launch.py','resume_evaluation.py'):shutil.copy2(p,dst/p.name)
shutil.copy2(BASE/'expanded_dynamics_v71/joint_data.py',dst/'joint_data.py')
for name in ('train_history.py','supervisor.py'):
    p=dst/name;s=p.read_text().replace('expanded_history_v76','full_history_v79')
    s=s.replace("source=ROOT/'run_goal_probability_v68'", "source=ROOT/'run_expanded_dynamics_v71'")
    s=s.replace('source_development_successes=44','source_development_successes=50')
    ast.parse(s);p.write_text(s)
print(dst)
