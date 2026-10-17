"""Regression guard: the evaluator must not grow the list it is iterating."""
import ast
from pathlib import Path
base=Path(__file__).parent.parent
for name in ('expanded_history_v76','stable_rules_v75','expanded_dynamics_v71','memory_dynamics_v77','joint_memory_dynamics_v78'):
    p=base/name/'evaluate.py';tree=ast.parse(p.read_text())
    for loop in ast.walk(tree):
        if not isinstance(loop,ast.For) or not isinstance(loop.iter,ast.Name):continue
        for stmt in loop.body:
            for item in ast.walk(stmt):
                if isinstance(item,ast.Call) and isinstance(item.func,ast.Attribute) and isinstance(item.func.value,ast.Name):
                    assert not(item.func.value.id==loop.iter.id and item.func.attr in ('append','extend','insert')), (name,loop.lineno,item.lineno)
    print(name,'iteration guard passed')
