"""Build isolated, self-contained contextual-symbol sources from v43."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'boundary_balanced_v43'
def change(s,old,new):
    assert s.count(old)==1,(old[:100],s.count(old))
    return s.replace(old,new)

model=(BASE/'model.py').read_text(encoding='utf-8')
model=change(model,'        self.detach_copy_evidence=True',
'''        # Entire 64x128 observation-definition vectors are generated at the
        # rule boundary. They are fixed for the lifetime of that program.
        self.observation_definition_generator=nn.Linear(c.context,c.observations*c.width)
        nn.init.zeros_(self.observation_definition_generator.weight)
        nn.init.zeros_(self.observation_definition_generator.bias)
        self.definition_feature_decoder=nn.Linear(c.width,49*128,bias=False)
        nn.init.normal_(self.definition_feature_decoder.weight,std=.01)
        self.detach_copy_evidence=True''')
start=model.index('    def observation_encoding(')
end=model.index('    def memory_step(',start)
model=model[:start]+'''    def observation_definition_delta(self, context):
        return self.observation_definition_generator(context).reshape(-1,self.cfg.observations,self.cfg.width)

    def observation_encoding(self, features, context=None):
        z=F.normalize(self.visual(features.float().flatten(-2)),dim=-1)
        if context is None:
            definitions=F.normalize(self.observation_definitions,dim=-1)
            logits=12*z@definitions.T
            selected=definitions[logits.detach().argmax(-1)]
        else:
            assert len(features)==len(context)
            definitions=F.normalize(self.observation_definitions[None]+self.observation_definition_delta(context),dim=-1)
            logits=12*torch.einsum('b...w,bow->b...o',z,definitions)
            discrete=F.one_hot(logits.detach().argmax(-1),self.cfg.observations).float()
            selected=torch.einsum('b...o,bow->b...w',discrete,definitions)
        return logits,z,selected

    def observe(self, features, context=None):
        return self.observation_encoding(features,context)[0]

    def definition_residual(self, observation, context):
        delta=torch.einsum('b...o,bow->b...w',observation,self.observation_definition_delta(context))
        return self.definition_feature_decoder(delta)

    def load_v43(self, state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        assert not unexpected
        assert set(missing)=={'observation_definition_generator.weight',
            'observation_definition_generator.bias','definition_feature_decoder.weight'}

'''+model[end:]
(HERE/'model.py').write_text(model,encoding='utf-8')

objective=(BASE/'objectives.py').read_text(encoding='utf-8')
objective=change(objective,"    observation_logits,z,zq=model.observation_encoding(x);observations=hard(observation_logits)",
'''    # Boundary generator uses only the real observed prefix. Its internal
    # global feature codes are NOT the symbols interpreted by the planner.
    base_logits,_,_=model.observation_encoding(x)
    base_observations=hard(base_logits)
    contexts=model.context_sequence(x,base_observations,previous,data['task'])
    context=contexts[index,start];definitions=model.definitions(context)
    # Current and future training labels share this one fixed definition set.
    observation_logits,z,zq=model.observation_encoding(x,context);observations=hard(observation_logits)''')
objective=change(objective,"    contexts=model.context_sequence(x,world_observations,previous,data['task'])\n    context=contexts[index,start];definitions=model.definitions(context)\n",'')
objective=change(objective,"    bo_logits,bz,bzq=model.observation_encoding(bx);bo=hard(bo_logits)",
'''    global_bo=hard(model.observe(bx))
    bc=model.context_sequence(bx,global_bo,bprevious,data['branch_task'])[bi,bs]
    bo_logits,bz,bzq=model.observation_encoding(bx,bc);bo=hard(bo_logits)''')
objective=change(objective,"    bc=model.context_sequence(bx,world_bo,bprevious,data['branch_task'])[bi,bs]\n",'')
objective=change(objective,"model.observation_encoding(successors)","model.observation_encoding(successors,bc)")
objective=change(objective,"model.reconstruct(observations[actual_mask])", "(model.reconstruct(observations)+model.definition_residual(observations,context))[actual_mask]")
objective=change(objective,"    reconstructed=model.state_reconstruct(joint_codes).reshape(-1,49,128)",
"    reconstructed=(model.state_reconstruct(joint_codes)+model.definition_residual(observations,context)[actual_mask]).reshape(-1,49,128)")
objective=change(objective,"    reconstructed_future=model.state_reconstruct(future_codes).reshape(-1,49,128)",
"    reconstructed_future=(model.state_reconstruct(future_codes)+model.definition_residual(so,bc)).reshape(-1,49,128)")
objective=change(objective,"    return total,metrics",'''    # Report the actual magnitude and within-program categorical usage, so a
    # new high prediction number cannot silently be explained by code collapse.
    delta=model.observation_definition_delta(context)
    metrics.update(definition_delta_rms=delta.detach().square().mean().sqrt(),
        local_observation_changes_from_global=(observation_logits.argmax(-1)[actual_mask]!=base_logits.argmax(-1)[actual_mask]).float().mean(),
        mean_codes_per_training_program=torch.stack([observations[b,:int(end[b])+1].detach().argmax(-1).unique().numel()*torch.ones((),device=x.device) for b in range(batch)]).mean())
    return total,metrics''')
(HERE/'objectives.py').write_text(objective,encoding='utf-8')

evaluate=(BASE/'evaluate.py').read_text(encoding='utf-8')
evaluate=change(evaluate,"programs={};memories={};predicted_obs={};ages={};", "programs={};memories={};predicted_obs={};ages={};program_contexts={};")
evaluate=change(evaluate,"                            features[i].append(raw[0]);observed=int(model.observe(raw).argmax(-1))\n                            obs_history[i].append(observed)",
'''                            features[i].append(raw[0])
                            observed_global=int(model.observe(raw).argmax(-1))
                            obs_history[i].append(observed_global)
                            observed=int(model.observe(raw,program_contexts.get(i)).argmax(-1))
                            observed_before_renewal=observed''')
evaluate=change(evaluate,"                                grounding_allowed[0]=True",'''                                context=model.context_sequence(history,obs,previous,task)[:,-1]
                                program_contexts[i]=context
                                local_logits,evidence,_=model.observation_encoding(history,context)
                                obs=hard(local_logits)
                                observed=int(local_logits[0,-1].argmax(-1))
                                grounding_allowed[0]=True''')
evaluate=change(evaluate,"                                    _,evidence,_=model.observation_encoding(history)\n",'')
evaluate=change(evaluate,"                                context=model.context_sequence(history,obs,previous,task)[:,-1]\n                                exported=", "                                exported=")
evaluate=change(evaluate,"records[i]['steps'].append(dict(step=step,memory=memories[i],observed=observed,action=action,", "records[i]['steps'].append(dict(step=step,memory=memories[i],observed=observed,observed_before_renewal=observed_before_renewal,action=action,")
# Prediction scoring must use the SAME definitions as the preceding prediction.
evaluate=evaluate.replace("b['observed']", "b['observed_before_renewal']")
(HERE/'evaluate.py').write_text(evaluate,encoding='utf-8')
for name in ('data.py','planner.py','program_archive.py'):
    (HERE/name).write_text((BASE/name).read_text(encoding='utf-8'),encoding='utf-8')
(HERE/'storage.py').write_text((BASE/'storage.py').read_text(encoding='utf-8').replace('run_boundary_balanced_v43','run_contextual_symbols_v48'),encoding='utf-8')

train=(BASE/'train.py').read_text(encoding='utf-8')
train=change(train,"model.initialize_v41(previous['model'])","model.load_v43(previous['model'])")
train=change(train,"        model.load_v43(previous['model'])", "        assert previous['manifest']['data']==corpus.manifest['data'], 'Training/validation split changed'\n        model.load_v43(previous['model'])")
train=train.replace("architecture='boundary_balanced_v43'","architecture='contextual_symbols_v48'")
train=train.replace("('grounded_memory_v42','boundary_balanced_v43')","('contextual_symbols_v48',)")
train=train.replace("optimizer_initialization='Full model, optimizer and per-rank RNG resume; only training position sampling changes from v42'", "optimizer_initialization='v43 best weights initialize existing modules; new full definition vectors zero-initialized; fresh AdamW'")
train=train.replace("observation_state='Frame-only categorical64, learned global definitions, no handcrafted semantics'", "observation_state='Categorical64 with full128-dimensional definitions generated from boundary history; fixed until rule renewal, no handcrafted semantics'")
train=train.replace("'Stable global definitions avoid a'", "'Boundary-frozen definitions avoid a'")
train=change(train,"    if rank==0:atomic(out/'manifest.json',manifest)",'''    manifest.update(observation_definitions_generated_in_full=True,
        observation_definitions_frozen_between_renewals=True,
        predicted_and_target_observations_use_same_program_definitions=True,
        memory_grounding_only_on_rule_renewal=True,
        source_model='v43 best3500; no Qwen vision substitution',
        frozen_feature_probe='Original Qwen vision and language hidden features did not improve held-out training inverse dynamics.',
        runtime_action_selection='Unchanged NumPy exhaustive probability derivation from exported integer table')
    if rank==0:atomic(out/'manifest.json',manifest)''')
(HERE/'train.py').write_text(train,encoding='utf-8')
print('Built standalone v48 model/objectives/runtime/train sources')
