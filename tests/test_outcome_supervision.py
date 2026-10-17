import pytest,torch
from nesy_vla.symbols import tokenize,DiscreteState
from nesy_vla.outcome_supervision import SymbolicExecutionDecoder,execution_loss


def test_execution_supervision_needs_hard_symbols_and_backpropagates():
    torch.manual_seed(42);decoder=SymbolicExecutionDecoder(16)
    definitions=torch.randn(8,16,requires_grad=True);observation=torch.randn(4,8,requires_grad=True)
    state=tokenize(observation.sigmoid());output=decoder(definitions,state)
    assert output.shape==(4,16,2)
    labels=torch.randint(2,output.shape).float();execution_loss(output,labels).backward()
    assert observation.grad.norm()>0 and definitions.grad.norm()>0
    with pytest.raises((ValueError,TypeError)):
        decoder(definitions,DiscreteState(state.codes,observation.sigmoid()))
    with pytest.raises(TypeError):decoder(definitions,observation)


def test_execution_decoder_is_invariant_to_symbol_relabeling():
    torch.manual_seed(13);decoder=SymbolicExecutionDecoder(16);definitions=torch.randn(8,16)
    state=tokenize(torch.rand(4,8));order=torch.randperm(8)
    other=DiscreteState(state.codes[:,order],state.values[:,order])
    assert torch.allclose(decoder(definitions,state),decoder(definitions[order],other),atol=1e-6)
    changed=DiscreteState(1-state.codes,1-state.values)
    assert not torch.allclose(decoder(definitions,state),decoder(definitions,changed))


def test_stop_has_no_synthetic_execution_label():
    decoder=SymbolicExecutionDecoder(16);state=tokenize(torch.rand(2,8))
    result=decoder(torch.randn(8,16),state);result.sum().backward()
    assert decoder.action.weight.grad[0].count_nonzero()==0
    assert decoder.action.weight.grad[1:].norm()>0


def test_outcome_attachment_rejects_split_history_and_image_changes(tmp_path):
    import hashlib,json
    from nesy_vla.outcome_supervision import attach_execution_outcomes
    images=tmp_path/'images';images.mkdir();folder=tmp_path/'labels';folder.mkdir()
    frames=[]
    for i in range(3):
        path=images/f'{i}.jpg';path.write_bytes(bytes([i]))
        frames.append(dict(image=path.name,action=[3,4,0][i],episode_id='e',_image_root=str(images),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),recovery_demonstration=True,supervision_mask=i>0))
    record=dict(image='1.jpg',context=['0.jpg','0.jpg','1.jpg'],history=['0.jpg','1.jpg'],previous_actions=[3],
        episode_id='e',observed_collision=[False]*17,execution_valid=[True]*17)
    def write():
        path=folder/'execution_outcomes.json';path.write_text(json.dumps([record]))
        (folder/'manifest.json').write_text(json.dumps(dict(outcomes_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            scene_audit=['train'],execution_outcome_rows=1)))
    def load():return attach_execution_outcomes(folder,{'e':{'scene':'train'}},{'e':frames},{'train'},images)
    write();assert load()['rows']==1 and len(frames[1]['execution_outcome_labels'])==16
    record['previous_actions']=[7];write()
    with pytest.raises(ValueError,match='history'):load()
    record['previous_actions']=[3];write()
    with pytest.raises(ValueError,match='split'):
        attach_execution_outcomes(folder,{'e':{'scene':'heldout'}},{'e':frames},{'train'},images)
    (images/'1.jpg').write_bytes(b'different')
    with pytest.raises(ValueError,match='checksum'):load()


def test_rule_cost_cannot_be_reduced_by_stopping_and_uses_actual_proof_distribution():
    from nesy_vla.outcome_supervision import rule_execution_cost
    torch.manual_seed(2);logits=torch.randn(3,17,requires_grad=True)
    labels=torch.zeros(3,16,2);labels[:,:5,0]=1;labels[:,:2,1]=1
    probabilities=logits.softmax(-1);cost=rule_execution_cost(probabilities,labels)
    rescaled=probabilities.clone();rescaled[:,1:]*=.01;rescaled[:,0]=1-rescaled[:,1:].sum(-1)
    assert torch.allclose(cost,rule_execution_cost(rescaled,labels),atol=1e-7)
    gradient=torch.autograd.grad(cost,logits)[0]
    assert gradient[:,0].abs().max()<1e-7
    assert gradient[:,1:3].min()>0 and gradient[:,6:].max()<0
    after=(logits-gradient).softmax(-1)
    assert rule_execution_cost(after,labels)<cost
