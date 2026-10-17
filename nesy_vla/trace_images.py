"""Byte-identical PNG trace images shared by hard links, without quality loss."""
import hashlib,io,os,time,uuid
from pathlib import Path


def save_lossless_image(image,destination,store=None):
    destination=Path(destination)
    # Pillow's encoder mutates encoderinfo on the supplied object. Independent
    # copies allow concurrent recording of the same already-loaded observation.
    image=image.copy()
    if store is None:
        image.save(destination,compress_level=3);return
    stream=io.BytesIO();image.save(stream,format='PNG',compress_level=3);data=stream.getvalue()
    digest=hashlib.sha256(data).hexdigest();folder=Path(store)/digest[:2];folder.mkdir(parents=True,exist_ok=True)
    blob=folder/(digest+'.png')
    for attempt in range(3):
        if not blob.exists():
            temporary=folder/f'{digest}.{os.getpid()}.{uuid.uuid4().hex}.tmp'
            try:
                with temporary.open('xb') as writer:writer.write(data);writer.flush();os.fsync(writer.fileno())
                try:os.link(temporary,blob)
                except FileExistsError:pass
            finally:temporary.unlink(missing_ok=True)
        try:
            os.link(blob,destination);return
        except FileNotFoundError:
            # A cold-store garbage collection may remove an unreferenced blob
            # just before the link. Re-publish it, preserving the same bytes.
            if not destination.parent.is_dir():raise
        except FileExistsError:
            if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:
                raise ValueError('A committed trace image cannot be overwritten with different pixels')
            return
    raise RuntimeError('Trace image store changed repeatedly during publication')


def collect_unreferenced_images(store,minimum_age=600):
    """Remove orphan store links only; live trace links retain their PNG bytes."""
    root=Path(store).resolve();now=time.time();removed=[]
    if not root.exists():return removed
    for path in root.glob('*/*.png'):
        digest=path.stem
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):continue
        if path.parent.name!=digest[:2] or path.is_symlink():continue
        try:
            stat=path.stat()
            if stat.st_nlink!=1 or now-stat.st_mtime<minimum_age:continue
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('Image content store integrity mismatch')
            path.unlink();removed.append(str(path))
        except FileNotFoundError:continue
    return removed
