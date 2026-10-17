import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
asset=ROOT/'assets/entmax_1_3';record=json.loads((asset/'provenance.json').read_text())
wheel=asset/record['filename'];assert hashlib.sha256(wheel.read_bytes()).hexdigest()==record['sha256']
# Only the stalled installer started by this experiment is eligible to stop.
for pid,identity in [(2058911,'-m pip install --disable-pip-version-check --no-deps --index-url https://mirrors.aliyun.com/pypi/simple --target '+str(ROOT/'vendor_entmax_1_3')+' entmax==1.3'),
                     (2058910,str(ROOT/'prepare_entmax_reference.py'))]:
    path=Path(f'/proc/{pid}/cmdline')
    if path.exists():
        command=path.read_bytes().replace(b'\0',b' ').decode()
        if command:
            assert identity in command, 'PID ownership changed; do not signal'
            os.kill(pid,signal.SIGTERM)
time.sleep(1)
subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps','--no-index',
    '--target',str(ROOT/'vendor_entmax_1_3'),str(wheel)],check=True)
(ROOT/'vendor_entmax_1_3/provenance.json').write_text(json.dumps(record,indent=2))
print(json.dumps(dict(installed=True,version='1.3',wheel_sha256=record['sha256'],isolated=True)))
