import hashlib,json
import pytest,torch
from nesy_vla.small_checkpoint import save_small_checkpoint

def test_roundtrip_preserves_optimizer_rng_and_commit_integrity(tmp_path):
    model=torch.nn.Linear(4,3);opt=torch.optim.AdamW(model.parameters());model(torch.randn(2,4)).sum().backward();opt.step()
    payload={'model':model.state_dict(),'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),'epoch':7}
    marker=save_small_checkpoint(tmp_path/'latest',payload,{'epoch':7,'optimizer_included':True},reservation_root=tmp_path,wait_seconds=0)
    path=tmp_path/'latest'/marker['checkpoint_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==marker['checkpoint_sha256']
    recovered=torch.load(path,weights_only=False)
    assert torch.equal(recovered['torch_rng'],payload['torch_rng'])
    for key,value in payload['model'].items():assert torch.equal(recovered['model'][key],value)
    for key,item in payload['optimizer']['state'].items():
        for name,value in item.items():assert torch.equal(recovered['optimizer']['state'][key][name],value)

def test_failed_reservation_keeps_previous_complete_checkpoint(tmp_path):
    first=save_small_checkpoint(tmp_path/'latest',{'a':torch.tensor([1])},{'epoch':1},reservation_root=tmp_path,wait_seconds=0)
    before=(tmp_path/'latest/COMMITTED').read_bytes()
    with pytest.raises(OSError):save_small_checkpoint(tmp_path/'latest',{'a':torch.tensor([2])},{'epoch':2},reservation_root=tmp_path,reserve_bytes=2**80,wait_seconds=0)
    assert (tmp_path/'latest/COMMITTED').read_bytes()==before
    assert torch.load(tmp_path/'latest'/first['checkpoint_file'],weights_only=True)['a'].item()==1

def test_writer_cannot_escape_experiment(tmp_path):
    with pytest.raises(ValueError):save_small_checkpoint(tmp_path.parent/'outside',{}, {},reservation_root=tmp_path,wait_seconds=0)
