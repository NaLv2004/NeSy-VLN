from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
def change(s,a,b):
    assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
s=(root/'train_evidence_rule_v23.py').read_text(encoding='utf-8')
s=change(s,'from predictive_nesy.evidence_rule_world import EvidenceRuleWorld as SubgoalRuleModel,Config',
    'from predictive_nesy.state512_rule_world import State512RuleWorld as SubgoalRuleModel,Config')
s=change(s,'from predictive_nesy.separated_goal_objectives import objective\nfrom predictive_nesy.separated_transition_objectives import transition_objective',
    'from predictive_nesy.selected_row_objectives import demonstration_objective,transition_objective')
s=change(s,'generated=self.model.generate(', 'generated=self.model.prepare(')
s=change(s,"if kind=='demonstrations':return objective(self.model,data,proof_from_predicted_state=True,program=generated)",
    "if kind=='demonstrations':return demonstration_objective(self.model,data,generated)")
s=change(s,'transition_objective(self.model,data,program=generated)','transition_objective(self.model,data,generated)')
s=change(s,'model=SubgoalRuleModel(Config()).to(device)','model=SubgoalRuleModel(Config(**original[\'model_config\'])).to(device)')
s=s.replace("'source_evidence_generated_world_v23'","'state512_source_evidence_world_v25'")
s=change(s,'learned_joint_state_count=128','learned_joint_state_count=512')
s=change(s,"initialization_scope='Complete v21step1000; zero-output source-conditioned visual attention initially preserves every generated program'",
    "initialization_scope='Complete v23step1000 components;512 learned visual definitions distilled into neural decoder; goal-head rows initialized by visual nearest-parent transfer'")
s=change(s,"world_model_intervention='Each source rule receives learned attention over current boundary image patches before the complete transition table is generated; no intermediate neural calls'",
    "world_model_intervention='512 anonymous visual states reduce label aliasing; observed-row gradients are algebraically equivalent to full-table supervision, and full generated rules are still exported at every revision'")
s=change(s,"monitor_only_losses=['dynamics','teacher_path_goal','rule_inverse','reconstruction','posterior_alignment']",
    "selected_row_supervision=True,full_rule_table_exported_at_runtime=True,monitor_only_losses=[]")
s=change(s,"'evaluate_evidence_rule_v23.py'","'evaluate_state512_v25.py'")
(root/'train_state512_v25.py').write_text(s,encoding='utf-8')
s=(root/'evaluate_evidence_rule_v23.py').read_text(encoding='utf-8')
s=change(s,'from predictive_nesy.evidence_rule_world import EvidenceRuleWorld as SubgoalRuleModel,Config,export_program',
    'from predictive_nesy.state512_rule_world import State512RuleWorld as SubgoalRuleModel,Config,export_program')
s=s.replace("'source_evidence_generated_world_v23'","'state512_source_evidence_world_v25'")
(root/'evaluate_state512_v25.py').write_text(s,encoding='utf-8')
print('Created512-state trainer and evaluator with full runtime programs and equivalent selected-row supervision.')
