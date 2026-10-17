import json,os,shutil,subprocess,time
from pathlib import Path
roots=[Path('/home/user/xjy/nesy_vla_20261010'),Path('/home/user/xjy/strips_wm_navigation_20261009')]
result={'time':time.time(),'free_gib':shutil.disk_usage(roots[0]).free/2**30,'folders':{},'large_files':[]}
for root in roots:
    result['folders'][str(root)]=subprocess.check_output(['du','-x','-d','2','-B1',str(root)],text=True)
    entries=[]
    for folder,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in ['.git','__pycache__','content_store']]
        for name in files:
            path=Path(folder)/name
            if not path.is_symlink() and path.stat().st_size>150*2**20:
                st=path.stat();entries.append({'path':str(path),'bytes':st.st_size,'inode':st.st_ino,'links':st.st_nlink})
    result['large_files']+=sorted(entries,key=lambda r:r['bytes'],reverse=True)
(roots[0]/'disk_inventory_20261011.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'time':result['time'],'free_gib':result['free_gib'],
    'folders':{r:sorted([(int(line.split()[0])/2**30,line.split(maxsplit=1)[1]) for line in v.splitlines()],reverse=True)[:15] for r,v in result['folders'].items()},
    'largest_files':result['large_files'][:20]},indent=2))
