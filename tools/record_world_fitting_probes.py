"""Record bounded probes as rejected, without promoting or restarting a run."""
import json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
def read(name):return json.loads((root/'evidence'/name).read_text(encoding='utf-8-sig'))
def compact(value):return {k:v for k,v in value.items() if k!='rows'}
p=json.loads((root/'goal_progress.json').read_text(encoding='utf-8'))
for name in ['v12_initial_on_v11_paths','v12_1600_on_v11_paths']:
    p[name]=compact(read(name+'.json')['results'][0])
for name in ['v12_1600_components','v12_1600_saved_navigation','v12_supervised_world_components']:
    p[name]=compact(read(name+'.json'))
p['v12_supervised_world_fidelity']=compact(read('v12_supervised_world_fidelity.json')['results'][0])
for name,file in [('v12_supervised_world_probe','v12_supervised_world_probe.json'),('v13_rule_context_probe','v13_rule_context_probe.json')]:
    r=read(file)
    r['promoted']=False;r['development_evaluation_run']=False
    r['initial_held_movement_accuracy']=sum(r['initial']['held']['per_action_accuracy'][1:])/16
    r['final_held_movement_accuracy']=sum(r['history'][-1]['held']['per_action_accuracy'][1:])/16
    r['decision']='Rejected: actual transition fitting improves seen rows and STOP consistency, but held movement dynamics deteriorate; loss reduction is not navigation improvement.'
    p[name]=r
p['v13_rule_context_probe_startup_repair']=dict(first_attempt='pilot_rule_context_v13',failed_before_optimizer_update=True,
    reason='cuDNN GRU backward requires training mode',repair='Enable training mode only for the independent rule-context branch; preserve all grounded-state and goal weights',
    repaired_attempt='pilot_rule_context_v13r1',completed=True)
p['current_turn_classification']='progress: v12 actual updates/development verified; completed-runtime state error geometry audited; two bounded world-fitting candidates rejected using transition and full-plan diagnostics'
p['last_record_update_unix']=time.time()
(root/'goal_progress.json').write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
path=root/'README.md';text=path.read_text(encoding='utf-8');marker='## Bounded world-fitting probes after v12'
if marker not in text:
    text+='\n'+marker+'\n\n'
    text+='A training-only probe cached8066 states and their17 observed action outcomes. Only the existing rule decoder was trained, with all grounded-state and goal tensors unchanged. An additional-fit split excluded1531 states by episode; the source model had already seen these banks, so this is not unseen navigation validation. In1500 updates, fit accuracy rose32.66% to59.27%, while excluded-episode accuracy ended26.81% (initial29.58%). The best overall excluded score31.14% came from STOP consistency61.01% to95.17%; movement accuracy27.62% to27.14% did not improve. On saved v11 paths, autonomous accuracy38/362 to51/362 and one-step91/362 to157/362 improved, but on128 demonstrated windows the first proven action fell28/128 to13/128 and74/128 had no proof. This candidate was not deployed or evaluated on development.\n\n'
    text+='A second bounded probe let an independent boundary-only neural context encoder learn the dynamics while preserving the complete original grounded-state and goal pipeline. Every rule parameter is still generated, and inference between rule revisions is unchanged. The rule-row shortcut matched full-table probabilities within4.2e-7, and changing padded later inputs did not change the selected context. The first attempt failed before an optimizer update because the GRU was in evaluation mode; enabling training mode for that branch repaired it. The completed1500-update probe fit63.79% but excluded-episode accuracy fell29.58% to23.98%. It was also rejected without a development rollout. These results support a representation/generalization limitation, not proof that longer training or loss reduction solves navigation.\n'
    path.write_text(text,encoding='utf-8')
print(json.dumps(dict(probes_recorded=True,both_rejected=True,goal_complete=False)))
