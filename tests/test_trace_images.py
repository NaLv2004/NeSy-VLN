import hashlib,os
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import pytest
from nesy_vla.trace_images import save_lossless_image


def test_identical_frames_share_storage_and_preserve_pixels(tmp_path):
    image=Image.new('RGB',(40,30),(91,27,199));store=tmp_path/'shared'
    paths=[tmp_path/f'{i}.png' for i in range(8)]
    with ThreadPoolExecutor(max_workers=4) as executor:list(executor.map(lambda path:save_lossless_image(image,path,store),paths))
    blobs=list(store.glob('*/*.png'));assert len(blobs)==1
    assert all(os.path.samefile(path,blobs[0]) for path in paths)
    assert Image.open(paths[0]).tobytes()==image.tobytes()
    assert not list(store.glob('*/*.tmp'))
    paths[0].unlink();assert Image.open(paths[1]).tobytes()==image.tobytes()


def test_different_pixels_are_distinct_and_existing_evidence_is_immutable(tmp_path):
    store=tmp_path/'shared';a=Image.new('RGB',(5,5),'red');b=Image.new('RGB',(5,5),'blue')
    save_lossless_image(a,tmp_path/'a.png',store);save_lossless_image(b,tmp_path/'b.png',store)
    assert len(list(store.glob('*/*.png')))==2
    before=(tmp_path/'a.png').read_bytes()
    with pytest.raises(ValueError):save_lossless_image(b,tmp_path/'a.png',store)
    assert (tmp_path/'a.png').read_bytes()==before


def test_default_and_shared_encoding_are_byte_identical(tmp_path):
    image=Image.new('RGB',(11,17),'orange')
    save_lossless_image(image,tmp_path/'normal.png');save_lossless_image(image,tmp_path/'linked.png',tmp_path/'store')
    assert (tmp_path/'normal.png').read_bytes()==(tmp_path/'linked.png').read_bytes()


def test_collection_preserves_live_and_recent_images(tmp_path):
    import time
    from nesy_vla.trace_images import collect_unreferenced_images
    store=tmp_path/'shared'
    for name,color in [('live','red'),('cold','blue'),('recent','green')]:
        save_lossless_image(Image.new('RGB',(8,8),color),tmp_path/f'{name}.png',store)
    old=time.time()-1200
    for name in ['live','cold']:os.utime(tmp_path/f'{name}.png',(old,old))
    (tmp_path/'cold.png').unlink();(tmp_path/'recent.png').unlink()
    removed=collect_unreferenced_images(store)
    assert len(removed)==1 and len(list(store.glob('*/*.png')))==2
    assert Image.open(tmp_path/'live.png').getpixel((0,0))==(255,0,0)
