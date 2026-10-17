"""Derive a separate anonymous-only collector without editing the live source."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
source=(root/'collect_policy_outcomes.py').read_text()
source=source.replace('from predictive_nesy.model_memory import PredictiveMemoryModel, MemoryConfig',
    'from predictive_nesy.anonymous_model import AnonymousRuleModel, Config')
source=source.replace('from predictive_nesy.interpreter import export_program',
    'from predictive_nesy.anonymous_interpreter import export_program, Controller')
source=source.replace('from predictive_nesy.belief_interpreter import BeliefController\n','')
source=source.replace("p.add_argument('--prepare-only',action='store_true')", "p.add_argument('--prepare-only',action='store_true')\n    p.add_argument('--seed',type=int,default=1527)")
source=source.replace('def select_training(bank, count):','def select_training(bank, count, seed=1527):').replace('random.Random(1527)','random.Random(seed)')
source=source.replace('select_training(bank,a.episodes)','select_training(bank,a.episodes,a.seed)')
source=source.replace("m=PredictiveMemoryModel(MemoryConfig(**saved['model_config'])).cuda().eval()",
    "if not saved['manifest']['architecture'].startswith('anonymous_generated_rules_'):raise ValueError('Anonymous checkpoint required')\n    m=AnonymousRuleModel(Config(**saved['model_config'])).cuda().eval()")
source=source.replace("features=['visual_patch_codes','executed_action','successor_visual_patch_codes','collision','goal_eligible']",
    "schema='anonymous_scalar_outcomes_v1',features=['visual_patch_codes','executed_action','successor_visual_patch_codes','scalar_task_score'],seed=a.seed")
source=source.replace('CollectorSimulator(bank,logs)',"CollectorSimulator(bank,logs,worker='anonymous_counterfactual_worker.py')")
source=source.replace('controller=BeliefController()','controller=Controller()')
source=source.replace("                collision=np.array([r['collision'] for r in trace],bool),\n                eligible_next=np.array([r['eligible_next'] for r in trace],bool),\n                eligible_current=np.array([r['eligible_current'] for r in trace],bool),",
    "                scalar_rewards=np.stack([r['scalar_rewards'] for r in trace]),\n                actual_rewards=np.array([r['scalar_rewards'][r['action']] for r in trace],np.float32),")
source=source.replace("forbidden_features_not_saved=['map','pose','goal_coordinates','distance']",
    "forbidden_features_not_saved=['map','pose','goal_coordinates','distance','collision_labels','goal_labels']")
(root/'collect_anonymous_outcomes.py').write_text(source,encoding='utf-8')
worker=(root/'counterfactual_worker.py').read_text()
worker=worker.replace("                    eligible=env.private_metrics()['eligible']\n                    images=[current];collisions=[False];goals=[eligible]",
    "                    scores=np.full(17,-.02,np.float32)\n                    scores[0]=1. if env.private_metrics()['eligible'] else -1.\n                    images=[current]")
worker=worker.replace("                        images.append(image(rgb));collisions.append(metrics['collision']);goals.append(metrics['eligible'])",
    "                        images.append(image(rgb))")
worker=worker.replace("                    agrees=bool(response.get('feedback',{}).get('collision',False))==bool(collisions[chosen])\n",'')
worker=worker.replace("result=dict(current=current,successors=np.stack(images),collision=collisions,eligible_next=goals,\n                        eligible_current=eligible,actual=actual,done=response['done'],probe_rgb_error=error,\n                        probe_matches_execution=bool(error<.5 and agrees),action=chosen)",
    "result=dict(current=current,successors=np.stack(images),scalar_rewards=scores,\n                        actual=actual,done=response['done'],probe_rgb_error=error,\n                        probe_matches_execution=bool(error<.5),action=chosen)")
(root/'anonymous_counterfactual_worker.py').write_text(worker,encoding='utf-8')
