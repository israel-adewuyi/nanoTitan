from abc import ABC, abstractmethod


class Runtime(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def prepare_model(self):
        pass

    @abstractmethod
    def log(self, step, values_to_log):
        pass

    @abstractmethod
    def prepare_trainloader(self, train_dataset):
        pass

    @abstractmethod
    def prepare_valloader(self, val_dataset):
        pass

    @abstractmethod
    def backward(self, loss):
        pass

    @abstractmethod
    def finalize_backward(self):
        pass

    @abstractmethod
    def cleanup(self):
        pass
