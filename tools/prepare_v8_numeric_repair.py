import hashlib,json,shutil
from pathlib import Path
root=Path('/home/user/xjy/nesy_vla_20261010/predictive_rules_20261015')
src=root/'src_rollout_proof_v8r1'
if src.exists():raise RuntimeError('Repair source already exists')
shutil.copytree(root/'src_rollout_proof_v8',src)
shutil.copyfile(root/'diagnostics/graph_subgoal_model_fp32.py',src/'predictive_nesy/graph_subgoal_model.py')
path=src/'train_rollout_proof.py';s=path.read_text()
s=s.replace("if __name__=='__main__':main()", """if __name__=='__main__':
    try:main()
    except Exception as error:
        import traceback
        if int(os.environ.get('RANK',0))==0 and '--out' in sys.argv:
            folder=Path(sys.argv[sys.argv.index('--out')+1])
            if folder.exists():atomic(folder/'status.json',dict(state='failed',pid=os.getpid(),time=time.time(),error=repr(error),traceback=traceback.format_exc(),goal_complete=False))
        raise""")
path.write_text(s)
print(json.dumps(dict(source=str(src),source_model_sha256=hashlib.sha256((src/'predictive_nesy/graph_subgoal_model.py').read_bytes()).hexdigest())))
