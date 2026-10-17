"""Atomic full-resume checkpoints with a size-based budget for small automata.

Shares a disk reservation lock among the new experiment's trainers. Does not
change the older large-model checkpoint writer or remove any other run's data.
"""
import fcntl,hashlib,io,json,os,shutil,time
from pathlib import Path
import torch

def _json(path,value):
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)

def save_small_checkpoint(folder,payload,metadata,*,reservation_root,reserve_bytes=512*2**20,wait_seconds=600):
    folder=Path(folder).resolve();reservation_root=Path(reservation_root).resolve()
    if not folder.is_relative_to(reservation_root):raise ValueError('Checkpoint must stay inside its experiment')
    folder.mkdir(parents=True,exist_ok=True)
    buffer=io.BytesIO();torch.save(payload,buffer);data=buffer.getvalue()
    if len(data)>256*2**20:raise ValueError('Use the large-model writer for this payload')
    required=reserve_bytes+2*len(data);lock=reservation_root/'.checkpoint_reservation.lock'
    while True:
        with lock.open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            free=shutil.disk_usage(reservation_root).free
            if free>=required:
                name=f'checkpoint.e{metadata.get("epoch",0)}.{time.time_ns()}.pt';target=folder/name
                temporary=target.with_suffix('.tmp')
                with temporary.open('wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
                digest=hashlib.sha256(data).hexdigest()
                verified=hashlib.sha256()
                with temporary.open('rb') as stream:
                    for block in iter(lambda:stream.read(8*2**20),b''):verified.update(block)
                assert verified.hexdigest()==digest
                os.replace(temporary,target)
                marker={'checkpoint_file':name,'checkpoint_sha256':digest,**metadata}
                _json(folder/'metadata.json',metadata);_json(folder/'COMMITTED',marker)
                directory_fd=os.open(folder,os.O_RDONLY)
                try:os.fsync(directory_fd)
                finally:os.close(directory_fd)
                alias=folder/f'checkpoint.alias.{os.getpid()}';alias.symlink_to(name);os.replace(alias,folder/'checkpoint.pt')
                waiting=folder/'WAITING_FOR_STORAGE.json'
                if waiting.exists():waiting.unlink()
                generations=sorted(folder.glob('checkpoint.*.pt'),key=lambda p:p.stat().st_mtime_ns,reverse=True)
                for obsolete in generations[2:]:
                    if obsolete.is_symlink() or not obsolete.resolve().is_relative_to(folder):raise ValueError('Unsafe obsolete generation')
                    obsolete.unlink()
                return marker
            _json(folder/'WAITING_FOR_STORAGE.json',{'time':time.time(),'required_free_bytes':required,'free_bytes':free,
                  'serialized_bytes':len(data),'epoch':metadata.get('epoch')})
        if not wait_seconds:raise OSError('Insufficient checkpoint reservation; prior commit preserved')
        time.sleep(wait_seconds)
