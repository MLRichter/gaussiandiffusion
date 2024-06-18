from typing import Protocol

class SNR_Scheduler(Protocol):

    iter: int

    def __init__(self):
        ...

    def step(self):
        ...

    def __call__(self, snr: float):
        ...

class ConstantSNR_Target(SNR_Scheduler):

    def __init__(self):
        self.iter = 0

    def step(self):
        self.iter += 1

    def __call__(self, snr: float):
        return snr


class LinearSNR_Target(SNR_Scheduler):

    def __init__(self, max_step: int, ascending: bool, min_snr: float = 1.0):
        self.ascending = ascending
        self.iter = 0
        self.max_step = max_step
        self.min_snr = min_snr

    def step(self):
        self.iter += 1

    @property
    def factor(self):
        if self.ascending:
            return min(self.iter / self.max_step, 1.0)
        else:
            return 1.0 - min(self.iter / self.max_step, 1.0)

    def __call__(self, snr: float):
        return self.min_snr + (snr-self.min_snr)*self.factor