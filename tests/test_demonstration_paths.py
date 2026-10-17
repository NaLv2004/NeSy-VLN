from pathlib import Path
import pytest
from nesy_vla.runtime import DataBank


def test_declared_demo_root_supports_lossless_dataset_and_rejects_escape(tmp_path):
    bank=object.__new__(DataBank);bank.demo_image_root=tmp_path.resolve()
    bank.allowed_image_roots={bank.demo_image_root}
    assert bank.demo_image_path({'image':'episodes/a/frame_000.png'})==tmp_path/'episodes/a/frame_000.png'
    with pytest.raises(ValueError):bank.demo_image_path({'image':'../validation/answer.png'})
    with pytest.raises(ValueError):bank.demo_image_path({'image':'answer.png','_image_root':str(tmp_path/'validation')})
