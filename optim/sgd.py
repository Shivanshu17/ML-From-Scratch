class SGD():
    def __init__(self, params, lr = 0.1, nesterov = False, momentum = 0, weight_decay = 0):
        if lr < 0.0:
            raise ValueError("Invalid Learning Rate")
        if momentum < 0:
            raise ValueError("Invalid momentum value")
        if weight_decay < 0:
            raise ValueError("Invalid weight decay value")
        defaults = dict(lr = lr, nesterov = nesterov, momentum = momentum, weight_decay = weight_decay)
    
    