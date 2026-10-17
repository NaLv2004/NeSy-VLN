# NeSy-VLA: task-conditioned dynamic predicate programs

This is the integrated implementation of `idea.md`. It generates continuous predicate definitions and typed soft rules with Qwen3.5-2B, grounds them in RGB and event history, executes a fixed differentiable logic engine, and revises the program after exactly 10 executed actions. A 32-action episode uses program versions generated at actions 0, 10, 20, and 30. No revision is performed after an episode has terminated.

## Execution and learning

- Qwen3.5-2B is loaded from the existing local remote-server model. Rank-8 LoRA and trainable query decoders produce 16 definitions of dimension 128 and 48 rules.
- Rule slots are typed structurally: 8 first-layer state rules, 8 second-layer state rules, 16 action rules, and 16 temporal-effect rules. No predicate meaning or navigation rule is supplied by hand.
- ConvNeXt-Tiny starts from the official ImageNet weights, not an earlier direct-action navigation head. A shared conditional predicate grounder attends to image tokens and up to 16 event tokens.
- Program revision receives chronological images, previous executable connections, executed actions, collision feedback and visual/state-prediction changes. The neural program decoder also reads the old program tensors. Changed definitions are regrounded rather than inheriting truth by slot index.
- Action probabilities are computed only by `SoftLogic.infer`. Auxiliary value and outcome heads cannot select actions. There is no RGB-to-action policy, neural reranker, map, coordinate input, goal-image oracle, or direct-action fallback.
- Each training update mixes actual closed-loop PPO rollouts, successful demonstration imitation, executed-action outcomes, EMA-grounded next-predicate targets, conflict/structure penalties, and periodic program-revision distillation.
- PPO uses one full on-policy pass, with ten-action truncated graphs and recorded past event memory. All gradients are accumulated before the synchronized optimizer update. This is not a claim of full-episode backpropagation through every historical image.
- Every fifth update, training-only paired program trials compare the incumbent, default revision and two perturbed revisions over six actions from the same reset/replayed state. The winning program supplies a distillation target. The main PPO rollout is not replaced by the privileged candidate winner. These short trials establish local return differences, not guaranteed long-horizon improvement.
- All six ranks run the complete model with eight parallel simulator trajectories per GPU. VLM requests are batched; all images in each training minibatch are encoded together with BF16 tensor-core convolutions. Predicate and soft-logic arithmetic stays FP32. Initial isolated capacity tests completed 14 ten-action chunks (22.39 GiB reserved, 19.88 GiB allocated); 16 and 18 overflowed. Those tests preceded the final checkpointing repair and did not include eight live renderers, so their limit is not treated as a universal live-training capacity. Gradient communication is coalesced. Checkpoints include trainable weights, EMA teacher, optimizer, and all six random states. Frozen base model weights are referenced, not duplicated.
- Successful demonstration replay, teacher encodings, and held-out evaluation use batched execution too. Each GPU adapts its training chunk count independently from a starting batch of 14, up to 32 shorter chunks when measured capacity permits. Initial and revision prompts are packed separately, with at most 130 action frames and 13 long-context revision chunks per minibatch. The capacity target leaves 3% physical memory for transient work and deducts simulator/driver memory. `batch_tuning_rank*.json` preserves the chosen batch across restarts; no dummy GPU allocation is used. Per-phase timing and actual frame counts are logged.
- Real six-GPU testing exposed a PyTorch non-reentrant checkpoint error after an OOM retry. The VLM now uses reentrant checkpointing with gradient-bearing text inputs. An OOM durably lowers the chunk/frame limits and exits the affected worker; the supervisor restores the entire six-rank optimizer/RNG state from the latest committed update in fresh processes. Partially accumulated gradients are discarded. This recovery can interrupt the current update, but never rolls back a committed update.

## Dataset boundary

Existing industrial-office training episodes and 1,951 successful demonstrations (17,560 action labels) are reused. Layouts 116, 117, 122 and 123 are internal development only. The original fixed 80-task set from layouts 200–207 remains evaluation-only, protected by SHA-256 checks. Candidate revisions and optimizer updates are prohibited on these layouts.

The policy is given the task, RGB, observed collision feedback and past actions. Simulator distance and target visibility remain outside the model; existing scalar rewards are training supervision. The physical simulator retains its collision machinery. The agent builds no spatial map.

Training and primary evaluation use at most 32 actions. Success remains the frozen environment's explicit STOP, distance at most 1 metre, and at least 16 visible target pixels. Older 64-action baseline scores must be labelled separately.

Every ten updates, distributed greedy evaluation runs once on development and fixed80. Model selection uses development only. `scripts/evaluate.py` additionally supports stochastic sampling and disabling revision for matched ablations. Evaluation never branches future simulator states, trains on outcomes, or carries learned programs between episodes.

## Run on the server

Project: `/home/user/xjy/nesy_vla_20261010`.

Python: `/home/user/xjy/objectnav_qwen35_rl_20260921/envs/policy/bin/python`.

```bash
python -m unittest discover -s tests -v
python scripts/train.py --out runs/integration --smoke --updates 1
python scripts/supervise.py --out runs/full_v1 --gpus 0,1,2,3,4,5
python scripts/evaluate.py --checkpoint runs/full_v1/latest --out runs/full_v1/eval_random --stochastic
python scripts/evaluate.py --checkpoint runs/full_v1/latest --out runs/full_v1/eval_no_revision --no-revision
```

`latest/COMMITTED` names and hashes the durable checkpoint generation. Save every update; keep the latest two generations and milestone snapshots every 25 updates. `best_development` is selected only on development episodes. Creating `STOP` inside a run directory asks its supervisor to stop its own process group.

## Evidence and limitations

`rank*.jsonl` records update time, peak GPU memory, module gradient norms, losses, training successes and paired revision trials. `evaluations/step_*/{development,fixed80}/summary.json` contains held-out success and collision rates by domain, category and distance.

Each trace directory contains observed images, full numeric program versions, the actual VLM prompts, per-rule action contributions, predicate states, prediction errors, a video and a browsable proof report. Latent predicate names are not presented as proven natural-language meanings. The architecture enforces the computational logic bottleneck; semantic interpretability and the 70% navigation target still require experimental evidence.

Unit checks cover action-proof equality, zero-support failure, nonconstant action premises, predicate permutation equivariance, gradient flow, old-program conditioning, unknown evidence, bounded state transitions, exactly ten-action revision scheduling, and private-feedback filtering. A real-VLM/real-simulator integration run verifies the entire update and checkpoint path.

The 2026-10-10 checks pass 14 unit tests, real sequential/batched inference comparisons across a revision boundary, and finite nonzero VLM gradients after switching from evaluation to reentrant-checkpoint training. Warmed two-demonstration replay took 0.93 s sequentially versus 0.73 s batched; this small test is not a whole-training speedup claim. Initial full-run update 10 took 44.6 s for 48 sampled episodes plus demonstration/revision learning, excluding checkpoint output and scheduled validation. Its first fixed80 result was 20/80 (25%); development was 9/80 (11.25%). These early results do not establish the 70% target. Evidence is saved under `output/` and remotely under `runs/full_v1/`.


## V2 anti-collapse implementation (2026-10-10)

`configs/navigation_v2.json` runs in the isolated remote `v2/` checkout; the original code, evaluation evidence and step-60 committed checkpoint remain under the parent project. V2 starts new symbolic/grounding parameters from the existing pretrained Qwen and ImageNet backbones rather than importing the collapsed v1 program.

V2 uses 32 learned predicate definitions and 96 typed rules. Learned orthogonal query initialization plus a bounded contextual definition residual prevents the initial definition slots from becoming copies; this supplies no navigation semantics. Each rule premise is a soft mixture of at most two learned literals, excluding a predicate/negation pair. A straight-through dense surrogate supplies connection gradients. Action-rule routing uses differentiable balanced matching, which allocates capacity across the enumerated motor actions without assigning predicate meanings. State-rule strengths are bounded to limit confidence inflation. Every autonomous action still comes exclusively from the original fixed soft-logic engine.

The joint objective adds predicate variance/decorrelation and inverse-action discrimination computed by comparing the symbolic transition predictions for the 17 possible actions. That auxiliary computation sees predicate states only and never selects a deployed action. The next-state target omits the just-executed action from its visual grounding context, preventing a target-memory shortcut. Demonstration and PPO losses are aggregated by action count instead of giving every short terminal chunk the same weight as ten actions. Candidate trials run to the remaining 32-action horizon; only successful, behaviorally distinct improvements can become distillation targets. Failed STOP candidates cannot supply targets.

The existing 10-action program revision schedule, scene split hashes, 32-action horizon and fixed success condition are unchanged. The trainer logs actual predicate-flip interventions, definition similarity and unanimous action-rule counts. Artificial-state sensitivity checks establish a functioning bottleneck, not learned semantics or navigation success. The 70% fixed80 objective remains unverified until measured in closed loop.
