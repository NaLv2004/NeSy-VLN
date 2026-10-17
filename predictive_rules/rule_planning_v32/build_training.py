from pathlib import Path
root=Path(__file__).resolve().parent
old=root.parent/'raw_grounding_v31'
text=(old/'train.py').read_text(encoding='utf-8')
def replace(a,b):
    global text
    if text.count(a)!=1:raise ValueError((a,text.count(a)))
    text=text.replace(a,b)
replace('from raw_objectives import demonstration_objective, branch_objective',
        'from raw_objectives import demonstration_objective, branch_objective\nfrom hard_planning_objective import planning_rule_objective')
replace("architecture='raw_observed_joint_v31'","architecture='raw_hard_planning_v32'")
replace("raw_successor_available=branches_bank.refresh())",
        "raw_successor_available=branches_bank.refresh(),\n        hard_planning_rule_gradients=True,hard_planning_weight=.25,hard_planning_batch_size=4,\n        hard_planning_forward='Same argmax successor and rounded confidence costs; zero-valued derivative surrogate only during training')")
replace("        branch_loss.backward();total_loss+=float(branch_loss.detach());metrics.update(branch_metrics)",
        "        branch_loss.backward();total_loss+=float(branch_loss.detach());metrics.update(branch_metrics)\n        # A small full-table gradient batch trains rule generation through the\n        # exact hard backward proof, without changing the runtime planner.\n        planning_rows=rows[:4];planning_data=batch(planning_rows,'cuda')\n        planning_program=prepare(model,planning_data,features,planning_rows)\n        with torch.autocast('cuda',dtype=torch.bfloat16):\n            planning_loss,planning_metrics=planning_rule_objective(model,planning_data,planning_program)\n        if not torch.isfinite(planning_loss):raise FloatingPointError(planning_metrics)\n        (.25*planning_loss).backward();total_loss+=.25*float(planning_loss.detach());metrics.update(planning_metrics)")
(root/'train.py').write_text(text,encoding='utf-8')
text=(old/'evaluate.py').read_text(encoding='utf-8')
replace("=='raw_observed_joint_v31'","=='raw_hard_planning_v32'")
(root/'evaluate.py').write_text(text,encoding='utf-8')
print('Prepared isolated hard-planning-gradient trainer.')
