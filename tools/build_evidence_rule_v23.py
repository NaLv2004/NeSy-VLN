"""New isolated source variants, preserving all v21 implementation files."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
def replace(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)
s=(root/'train_separated_goal_v21.py').read_text(encoding='utf-8')
s=replace(s,'from predictive_nesy.separated_goal_world import SeparatedGoalWorld as SubgoalRuleModel,Config',
    'from predictive_nesy.evidence_rule_world import EvidenceRuleWorld as SubgoalRuleModel,Config')
s=replace(s,"    for target_name in list(current):\n        if target_name.startswith('goal_'):\n            source_name='raw_'+target_name[len('goal_'):]\n            if source_name in original['model']:current[target_name]=original['model'][source_name].clone()\n",'')
s=s.replace("('definition_rules.','subgoal.','raw_','goal_')","('definition_rules.','subgoal.','raw_','goal_','rule_evidence.')")
s=s.replace("('definition_rules.','raw_')","('definition_rules.','raw_','rule_evidence.')")
s=s.replace("'separated_rule_goal_world_v21'","'source_evidence_generated_world_v23'")
s=replace(s,"initialization_scope='Complete v20step600; duplicate its raw branch into an independent goal branch with an initially identical program'",
    "initialization_scope='Complete v21step1000; zero-output source-conditioned visual attention initially preserves every generated program'")
s=replace(s,"world_model_intervention='Observed one-step transitions train complete rules; independent goal branch receives endpoint and proof-action supervision; proof gradients cannot modify rules'",
    "world_model_intervention='Each source rule receives learned attention over current boundary image patches before the complete transition table is generated; no intermediate neural calls'")
s=replace(s,"'goal_history','goal_delta'],state_representation_frozen=True,","'goal_history','goal_delta','rule_evidence'],state_representation_frozen=True,")
s=replace(s,"'evaluate_separated_goal_v21.py'","'evaluate_evidence_rule_v23.py'")
(root/'train_evidence_rule_v23.py').write_text(s,encoding='utf-8')
s=(root/'evaluate_separated_goal_v21.py').read_text(encoding='utf-8')
s=replace(s,'from predictive_nesy.separated_goal_world import SeparatedGoalWorld as SubgoalRuleModel,Config,export_program',
    'from predictive_nesy.evidence_rule_world import EvidenceRuleWorld as SubgoalRuleModel,Config,export_program')
s=s.replace("'separated_rule_goal_world_v21'","'source_evidence_generated_world_v23'")
(root/'evaluate_evidence_rule_v23.py').write_text(s,encoding='utf-8')
print('Created isolated v23 trainer and evaluator; unchanged losses, data sampling, split and exact-goal interpreter.')
