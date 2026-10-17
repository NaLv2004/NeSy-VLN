"""Create a separate model that has no reward, value, or action-score head."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules/predictive_nesy'
source=(root/'anonymous_model.py').read_text()
source=source[:source.index('\ndef parent_joint')]
source=source.replace('Anonymous categorical dynamics; no manually named semantic state heads.',
    'Anonymous generated dynamics and a learned future symbolic subgoal.')
source=source.replace('class AnonymousRuleModel(nn.Module):','class SubgoalRuleModel(nn.Module):')
source=source.replace("execution_schema='anonymous_v1'","execution_schema='subgoal_v1'")
source=source.replace("        self.rewards=RowDecoder([p,c.actions,c.states,c.states],c.hidden,1)\n        self.values=RowDecoder([p,c.states,c.states],c.hidden,1)\n        self.rule_weights=nn.Linear(c.hidden,p)",
    "        self.subgoal=RowDecoder([c.factors],c.hidden,c.states)")
source=source.replace("            rewards=self.rewards(context).float().squeeze(-1).tanh(),\n            values=self.values(context).float().squeeze(-1).tanh(),\n            weights=self.rule_weights(context).float().softmax(-1),start=start,context_raw=raw)",
    "            subgoal_logits=self.subgoal(context).float(),start=start,context_raw=raw)")
source=source.replace('self.pairs=list(combinations(range(c.factors),2));p=len(self.pairs)','')
(root/'subgoal_model.py').write_text(source,encoding='utf-8')
