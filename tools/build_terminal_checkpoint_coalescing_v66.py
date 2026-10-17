from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'scripts/coalesce_retired_final_snapshots_v37.py').read_text()
s=s.replace("audit=base/'retired_final_coalescing_v37.json'", "audit=base/'terminal_automaton_coalescing_v66.json'\nassert not audit.exists(), 'Inspect existing completed cleanup audit'")
s=s.replace("['symbolic_teacher_v25','symbolic_task_v15','symbolic_branch_v17','symbolic_conjunctive_v23']", "['factorized_automaton_v51','goal_conditioned_automaton_v55','probabilistic_automaton_v56','higher_order_capacity_v64']")
s=s.replace("(base/family/'runs').glob('*')", "(base/family).glob('*')")
s=s.replace("for folder in run.glob('step_*'):", "for folder in [*run.glob('step_*'),run/'best_development']:")
(root/'scripts/coalesce_terminal_automata_v66.py').write_text(s)
