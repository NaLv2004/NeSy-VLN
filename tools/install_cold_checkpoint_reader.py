import ast,hashlib,json,shutil,time
from pathlib import Path
root=Path('/home/user/xjy/strips_wm_navigation_20261009')
path=root/'stripsnav/io.py';incoming=root/'cold_io_proposed.py'
old=path.read_text();new=incoming.read_text()
oldtree=ast.parse(old);newtree=ast.parse(new)
functions={n.name:n for n in newtree.body if isinstance(n,ast.FunctionDef)}
target=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name=='load_committed')
assert 'load_archived_committed' not in old,'Reader already installed; inspect before retry'
replacement=ast.get_source_segment(new,functions['load_committed'])+'\n\n\n'+ast.get_source_segment(new,functions['load_archived_committed'])+'\n'
lines=old.splitlines(keepends=True);result=''.join(lines[:target.lineno-1])+replacement+''.join(lines[target.end_lineno:])
ast.parse(result)
backup=path.with_name('io_before_cold_archive.py');assert not backup.exists();shutil.copy2(path,backup)
temporary=path.with_suffix('.tmp');temporary.write_text(result);temporary.replace(path)
print(json.dumps({'time':time.time(),'backup':str(backup),'old_sha256':hashlib.sha256(old.encode()).hexdigest(),'new_sha256':hashlib.sha256(result.encode()).hexdigest()}))
