import typing as tp
import numpy as np
import numpy.typing as npt


class Step:
    def __init__(self, method: tp.Callable):
        self.method = method
    
    def __call__(self, *args, **kwargs):
        return self.method(*args, **kwargs)


def numpy_wrap(data):
    if isinstance(data, list) and not isinstance(data, np.ndarray):
        if isinstance(data[0], np.ndarray):
            return np.array(data)
    return data


class Map:
    def __init__(self, method: tp.Callable):
        self.method = lambda x: list(map(method, x))
    
    def __call__(self, *args, **kwds):
        return self.method(*args, **kwds)

class StepMap(Step):
    def __init__(self, method: tp.Callable):
        super().__init__(lambda x: list(map(method, x)))

class Model:
    def __init__(self) -> None:
        pass

    def _forward_impl(self, data, step_mode=True):
        assert "Not implemented"

    def forward(self, data):
        return self.forward_impl(data, step_mode=False)
    
    def forward_step(self, data):
        return self.forward_impl(data, step_mode=True)

    

class Sequential(Model):
    def __init__(self, modules: tp.List[tp.Callable]):
        super().__init__()
        self.modules = modules + [numpy_wrap]
    
    def forward(self, data):
        for module in self.modules:
            if isinstance(module, Step):
                continue
            data = module(data)
        return data
    
    def forward_step(self, data):
        for module in self.modules:
            data = module(data)
        return data