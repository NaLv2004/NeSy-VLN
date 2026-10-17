import torch
from nesy_vla.perception import EvidencePredicateQuery,image_grid_encoding


def test_image_positions_are_distinct_centered_and_finite():
    positions=image_grid_encoding(7,128)
    assert positions.shape==(49,128)
    assert torch.isfinite(positions).all()
    assert torch.allclose(positions.mean(0),torch.zeros(128),atol=1e-5)
    assert abs(float(positions.square().mean())-1.)<1e-5
    assert len(torch.unique(positions,dim=0))==49


def test_missing_evidence_cannot_pass_definition_residual():
    torch.manual_seed(12)
    query=EvidencePredicateQuery(32)
    definitions=torch.randn(2,8,32)
    output=query(definitions,torch.zeros(2,49,32))
    assert torch.equal(output,torch.zeros_like(output))


def test_definitions_select_evidence_and_both_receive_gradient():
    torch.manual_seed(12)
    query=EvidencePredicateQuery(32)
    definitions=torch.randn(2,8,32,requires_grad=True)
    evidence=torch.randn(2,49,32,requires_grad=True)
    output=query(definitions,evidence)
    other=query(definitions,evidence.flip(0))
    assert not torch.allclose(output,other)
    output.square().mean().backward()
    assert definitions.grad.norm()>0
    assert evidence.grad.norm()>0
    # Reordering predicate definitions only reorders predicate observations.
    permutation=torch.randperm(8)
    assert torch.allclose(query(definitions[:,permutation],evidence),output[:,permutation],atol=1e-5)
