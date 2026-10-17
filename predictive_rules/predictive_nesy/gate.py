"""Runtime enforcement of the observation boundary, including nested modules."""
from contextlib import contextmanager


class NeuralBoundaryGate:
    def __init__(self, modules, interval=10):
        self.interval = interval
        self.allowed = False
        self.step = None
        self.calls = 0
        self.violations = 0
        self.handles = []
        for root in modules:
            for module in root.modules():
                self.handles.append(module.register_forward_pre_hook(self.check))

    def check(self, module, inputs):
        if not self.allowed:
            self.violations += 1
            raise RuntimeError('Neural inference attempted between rule revisions')
        self.calls += 1

    @contextmanager
    def boundary(self, step):
        if self.allowed or step < 0 or step % self.interval:
            raise RuntimeError('Observation correction is permitted only every ten executed actions')
        self.allowed, self.step = True, step
        try:
            yield
        finally:
            self.allowed, self.step = False, None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
