import json,shutil,sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,'/home/user/xjy/strips_wm_navigation_20261009')
from stripsnav import io


def test_disk_wait_preserves_old_commit_until_atomic_save_can_finish(tmp_path,monkeypatch):
    folder=tmp_path/'checkpoint'
    # The first tiny fixture save must not wait on the host's real free space.
    # The subsequent fake low-space phase exercises the guard explicitly.
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=10*2**30))
    io.committed_torch_save(folder,{'weights':torch.tensor([1.,2.])},{'epoch':1})
    marker=(folder/'COMMITTED').read_bytes();old=json.loads(marker)
    old_file=folder/old['checkpoint_file'];original=old_file.read_bytes();resumed=[]
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=(10*2**30 if resumed else 0)))
    def restore_capacity(seconds):
        assert seconds==60 and (folder/'COMMITTED').read_bytes()==marker
        assert old_file.read_bytes()==original and not list(folder.glob('checkpoint.*.tmp'))
        assert json.loads((folder/'WAITING_FOR_STORAGE.json').read_text())['epoch']==2
        resumed.append(True)
    monkeypatch.setattr(io.time,'sleep',restore_capacity)
    io.committed_torch_save(folder,{'weights':torch.tensor([3.,4.])},{'epoch':2})
    assert resumed==[True] and not (folder/'WAITING_FOR_STORAGE.json').exists()
    assert torch.equal(io.load_committed(folder)['weights'],torch.tensor([3.,4.]))
    assert old_file.read_bytes()==original
