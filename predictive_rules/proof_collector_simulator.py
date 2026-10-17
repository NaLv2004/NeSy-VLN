"""IPC for private visual-only transition probes, independent of policy input."""
import os,select,subprocess,sys
from pathlib import Path


class CollectorSimulator:
    def __init__(self,bank,out,worker='proof_counterfactual_worker.py'):
        if worker!='proof_counterfactual_worker.py':raise ValueError('Visual-only worker required')
        sys.path.insert(0,'/home/user/xjy/objectnav_qwen35_rl_20260921/src_domain_sft_v1')
        from objectnav_rl.ipc import receive,send
        self.receive,self.send=receive,send;self.log=(out/'simulator.log').open('ab')
        env=dict(os.environ,MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        self.process=subprocess.Popen([bank.env['sim_python'],str(Path(__file__).with_name(worker))],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log)
        self.call('init',experiment=bank.env,train_scenes=bank.provenance['train_layouts'])
    def call(self,operation,**kwargs):
        self.send(self.process.stdin,dict(op=operation,**kwargs))
        if not select.select([self.process.stdout],[],[],180)[0]:raise TimeoutError('Observe same collector handle after timeout; never duplicate')
        value=self.receive(self.process.stdout)
        if not value['ok']:raise RuntimeError(value['error'])
        return value['result']
    def close(self):
        if self.process.poll() is None:
            try:self.call('close')
            finally:self.process.wait(timeout=15)
        self.log.close()
