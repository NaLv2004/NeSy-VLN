"""Small, resumable foundation-encoder asset fetch into the isolated root."""
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import requests

ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
if os.name=='nt':assert os.environ.get('DINO_ASSET_DIR'),'Windows staging requires an explicit directory'
out=Path(os.environ.get('DINO_ASSET_DIR',str(ROOT/'assets/dinov2_small')));out.mkdir(exist_ok=True,parents=True)
session=requests.Session();session.trust_env=False
if os.name=='nt':
    class DirectCampusAdapter(requests.adapters.HTTPAdapter):
        def init_poolmanager(self,*args,**kwargs):
            kwargs['source_address']=(os.environ['STRIPS_SSH_LOCAL_IP'],0)
            kwargs['socket_options']=[(socket.IPPROTO_IP,31,socket.htonl(int(os.environ.get('STRIPS_SSH_INTERFACE','10'))))]
            return super().init_poolmanager(*args,**kwargs)
    session.mount('https://',DirectCampusAdapter())
origins=('https://hf-mirror.com','https://huggingface.co')
metadata=None
for origin in origins:
    try:
        response=session.get(origin+'/api/models/facebook/dinov2-small',params={'blobs':'true'},timeout=(5,15))
        response.raise_for_status();metadata=response.json();break
    except (requests.RequestException,ValueError) as e:print(type(e).__name__,str(e)[:160],flush=True)
assert metadata and metadata.get('sha')
revision=metadata['sha'];siblings={r['rfilename']:r for r in metadata['siblings']}
weight='model.safetensors' if 'model.safetensors' in siblings else 'pytorch_model.bin'
files=['config.json','preprocessor_config.json',weight]
provenance=dict(repository='facebook/dinov2-small',revision=revision,origin=origin,
    purpose='Candidate visual-feature encoder only; no neural action or planning output',
    proxy_environment_used=False,files=[])
for name in files:
    assert name in siblings
    info=siblings[name];expected=info.get('lfs',{}).get('sha256');size=info.get('size') or info.get('lfs',{}).get('size')
    path=out/name;partial=path.with_suffix(path.suffix+'.part');started=time.time()
    if path.exists():
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if (not expected or digest==expected) and (not size or path.stat().st_size==size):
            provenance['files'].append(dict(name=name,bytes=path.stat().st_size,sha256=digest,reused=True));continue
        raise RuntimeError(f'Existing completed file failed its metadata check: {name}')
    for attempt in range(12):
        origin=origins[min(attempt//6,1)]
        url=origin+f'/facebook/dinov2-small/resolve/{revision}/{name}'
        offset=partial.stat().st_size if partial.exists() else 0
        try:
            with session.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(5,20)) as response:
                response.raise_for_status()
                if offset and response.status_code==206:
                    assert response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
                    mode='ab'
                else:mode='wb'
                with partial.open(mode) as f:
                    for chunk in response.iter_content(1<<20):
                        if chunk:f.write(chunk)
            digest=hashlib.sha256(partial.read_bytes()).hexdigest()
            assert not expected or digest==expected, f'{name}: checksum mismatch'
            assert not size or partial.stat().st_size==size, f'{name}: size mismatch'
            partial.replace(path)
            row=dict(name=name,bytes=path.stat().st_size,sha256=digest,expected_sha256=expected,
                seconds=time.time()-started,download_origin=origin,reused=False)
            provenance['files'].append(row);print(json.dumps(row),flush=True);break
        except (requests.RequestException,AssertionError) as e:
            print(json.dumps(dict(file=name,attempt=attempt,error=str(e)[:160])),flush=True)
            if attempt==11:raise
            time.sleep(1)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
print(json.dumps(dict(complete=True,path=str(out),revision=revision,total_bytes=sum(r['bytes'] for r in provenance['files']))))
