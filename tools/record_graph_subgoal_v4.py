"""Record the measured architecture correction without altering archived runs."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'predictive_rules'
remote='/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015'
progress=json.loads((root/'goal_progress.json').read_text(encoding='utf-8'))
progress.update(remote_run=remote+'/run_graph_subgoal_v4',remote_source=remote+'/src_graph_subgoal_v4',
    last_verified_pid=532501,last_verified_update=141,last_remote_check_unix=1792089650.7519794,
    next_regular_check_not_before_unix=1792090251,contract_tests={'passed':9,'total':9},
    pending=['Measure genuine development navigation success after exact training/execution alignment',
             'Reach the unchanged reference success counts; do not equate valid proofs with correct learned dynamics',
             'Keep final fixed80 isolated; report inference time separately from neural-call frequency'])
progress['active_runs']['subgoal_v1'].update(last_observed_update=6000,last_observed_state='training_budget_complete',
    final_development={'successes':0,'episodes':80},best_development={'update':1600,'successes':1,'episodes':80},superseded=True)
progress['active_runs']['graph_subgoal_v3']=dict(last_observed_state='retired_planner_mismatch',last_observed_update=1367,
    preserved_update=1300,checkpoint_sha256='6469ea4582a48c5aef63da634d4d6d07b3d5ca8dd34c673219a1745836a40fe1',superseded=True,
    training_diagnostic={'samples':128,'relaxed_execution_agreement':0.,'immediate_stop':126,'demonstrated_immediate_stop':8})
progress['active_runs']['graph_subgoal_v4']=dict(path=remote+'/run_graph_subgoal_v4',pid=532494,train_gpus=[3,4,5],evaluation_gpu=7,
    last_observed_state='training',last_observed_update=141,actual_optimizer_updates_confirmed=True,
    latest_checkpoint={'update':100,'sha256':'1071cbc11e3a7d48d536af793480e82001ae23517515ec43b59df2333034becc','hash_verified':True},
    recent_mean_update_seconds=.3420714354515076,global_batch_size=1920,per_card_batch_size=640,
    peak_allocated_gib=13.845270156860352,nvidia_smi_used_mib=[20856,20833,20834],
    initial_source=remote+'/run_graph_subgoal_v3/checkpoint_000400.pt',fresh_optimizer=True,
    corrected_training_execution_agreement={'samples':128,'agreement':1.0,'checkpoint':'graph_v3_step400','navigation_improvement_established':False},
    mixed_precision_exact_agreement={'compared_valid_proofs':24,'passed':True},validation=[])
progress['previous_completed_development']={'run':'subgoal_v1','update':6000,'successes':0,'episodes':80}
(root/'goal_progress.json').write_text(json.dumps(progress,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
readme=(root/'README.md').read_text(encoding='utf-8')
begin=readme.index('## Current implementation:');end=readme.index('## Historical implementation:',begin)
section='''## Current implementation: generated discrete subgoals, graph v4

The active source is `src_graph_subgoal_v4`; its isolated output is
`run_graph_subgoal_v4`. Reward/value-table planners are retired. The generator
now produces all anonymous state definitions, every transition parameter,
and one discrete subgoal. No manually named collision, goal or progress state,
semantic auxiliary classifier, action head, learned reward or value table is
present in this model. Training uses GPUs 3/4/5; development evaluation uses GPU7.
Unrelated processes on the other cards remain running.

### What happens during execution

1. Only at executed-action indices 0, 10, 20, 30, the neural generator reads
   the task, up to 11 recorded RGB frames, and preceding executed actions.
   The frozen visual frontend produces 49 learned visual IDs per image.
   These are observations, not the symbolic navigation state.
2. A trainable history encoder generates 128 state-definition vectors and
   the entire 128 x 17 x 128 transition distribution. Their IDs have no
   manually assigned meanings. It also generates 128 subgoal logits.
3. The boundary observation is grounded into one anonymous current-state ID.
   The generated goal is its own argmax ID, unchanged even if unreachable.
4. The symbolic interpreter selects the successor ID of every state/action
   rule and searches the full finite-state system for a sequence ending
   exactly at that goal. A block has up to ten actions; the final block has
   two. Existing motor action STOP may close a proof early.
5. Every executed action belongs to that complete proof. Intermediate states
   come from table lookup alone. During a block the controller accepts no
   images, position, distances, collision feedback, or neural-network calls.
   New observations may be used only at the next ten-action rule revision.
6. If the generated goal is unreachable, execution reports planning failure.
   It does not substitute a reachable goal or use an unproven action fallback.

The finite-state search uses transition confidence to choose between valid
proofs. It never assigns learned utility to a state. Rule log probabilities
are quantized at 0.0001 for reproducible tie handling; equal confidence favors
fewer actions. This is a numerical convention, not an environment reward.
The total episode limit remains 32 actions.

### What is learned

The existing training-only corpus contains 2,578 demonstration/recovery
sequences and 3,680 boundary windows. Definitions and grounding are trained
jointly from visual reconstruction and action-conditioned state prediction.
The desired subgoal label is the learned encoding of the demonstrated future
state after up to ten actions. Future images are supervision only; they are
never generator inputs. Recovery-prefix actions marked unsuccessful are not
imitated. Their observed state changes can still supervise dynamics.

The proof-derived action loss supervises the action sequence produced by the
generated rules and generated goal. There is no separate neural action head.
The training forward pass uses the same discrete successor choices and proof
confidence as the executable NumPy planner. Straight-through derivatives
are a gradient surrogate, not a second execution procedure. Finite support
for an impossible proof is a derived numerical training bound; execution
continues to reject every impossible proof.

The model is initialized from graph-v3 step400; the optimizer is new. All
training layouts and both held-out suites are unchanged. Development80 is
used for model selection; independent fixed80 is reserved for final comparison.

### Verified evidence and current limits

- The first factored-subgoal run completed6,000 updates. Best development
  success was1/80; final success was0/80. A separate inverse-action trial
  also achieved0/80. These runs do not satisfy the requested performance.
- Graph-v3 was retired at committed step1,300. On128 training examples its
  relaxed planner and executable planner agreed on the first action0% of
  the time. The latter immediately stopped126 times; demonstrations stopped8.
- Graph-v4 corrects that forward mismatch. On the same frozen checkpoint and
  128 examples, agreement is100%. This does not prove navigation improvement.
- Nine contract tests passed. GPU BF16/TF32 execution also matched all24
  valid proofs checked. A real-data forward/backward preflight was finite.
- At02:40 on October16, v4 had actually completed141 optimizer updates.
  Recent mean compute time was0.342 seconds/update, with1,920 demonstration
  windows globally and32 transition examples per card. Observed allocation
  peak was13.85GiB/card and nvidia-smi usage about20.8GiB/card.
- Saves occur every100 updates; development navigation every400. Stable
  external status checks are at least600 seconds apart. No v4 success rate
  was available at that startup check.

Each evaluation exports its rules, goals, complete plans and executed proof
steps. An independent trace replay and a neural-call boundary gate check the
execution contract. Proof validity alone is insufficient: the learned rule
predictions must also match what the environment actually does.

Neural calls occur only at boundaries, but boundary processing re-encodes up
to11 recorded images. Consequently this design does not establish a10x visual
compute saving. Inference latency and navigation success must be measured.

'''
(root/'README.md').write_text(readme[:begin]+section+readme[end:],encoding='utf-8')
print(json.dumps({'recorded':'graph_subgoal_v4','verified_update':141,'goal_complete':False}))
