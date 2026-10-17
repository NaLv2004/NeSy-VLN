import hashlib,json,os,socket
from pathlib import Path
import requests
class DirectAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self,*args,**kwargs):
        kwargs['source_address']=(os.environ['STRIPS_SSH_LOCAL_IP'],0)
        kwargs['socket_options']=[(socket.IPPROTO_IP,31,socket.htonl(int(os.environ['STRIPS_SSH_INTERFACE'])))]
        return super().init_poolmanager(*args,**kwargs)
session=requests.Session();session.trust_env=False;session.mount('https://',DirectAdapter());proxy_used=False
try:
    response=session.get('https://pypi.org/pypi/entmax/1.3/json',timeout=(3,8));response.raise_for_status()
except requests.RequestException:
    session.close();session=requests.Session();session.trust_env=False
    session.proxies={'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'};proxy_used=True
    response=session.get('https://pypi.org/pypi/entmax/1.3/json',timeout=(5,15));response.raise_for_status()
record=next(r for r in response.json()['urls'] if r['filename'].endswith('.whl'))
response=session.get(record['url'],timeout=(5,15));response.raise_for_status()
digest=hashlib.sha256(response.content).hexdigest();assert digest==record['digests']['sha256']
out=Path(__file__).resolve().parent/'reference_asset';out.mkdir(exist_ok=True)
(out/record['filename']).write_bytes(response.content)
(out/'provenance.json').write_text(json.dumps(dict(upstream='https://github.com/deep-spin/entmax',
    version='1.3',filename=record['filename'],sha256=digest,download_url=record['url'],bytes=len(response.content),
    proxy_used=proxy_used),indent=2))
print(json.dumps(dict(downloaded=True,name=record['filename'],sha256=digest,bytes=len(response.content))))
