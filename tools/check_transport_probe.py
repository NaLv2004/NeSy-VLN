import json,time
from pathlib import Path
out=Path('/home/user/xjy/nesy_vla_20261010/symbolic_sft_v5/diagnostics/plain30_transport_development')
result={key:json.loads((out/f'{key}.json').read_text()) if (out/f'{key}.json').exists() else None
    for key in ['process','progress','summary']}
pid=result['process']['pid'];handle=Path(f'/proc/{pid}/cmdline')
result['live_command']=handle.read_bytes().replace(b'\0',b' ').decode() if handle.exists() else None
if result['summary'] is None and result['live_command'] is None:
    result['error_tail']=(out/'process.log').read_text(errors='replace')[-3500:]
result['time']=time.time();print(json.dumps(result,indent=2))
