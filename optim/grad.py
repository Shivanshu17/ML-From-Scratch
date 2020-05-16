class grad():
    def __init__(self, cost = 0, data, params):
        self.cost = cost
        self.data = data
        self.params = params
        
    def grad_mse(self):
        
    
    if __name__ == "__main__":
        if cost == 0:
            gradient = grad_mse()
        if cost == 1:
            gradient = grad_mae()
        if cost == 2:
            gradient = grad_huber()
        if cost == 3:
            gradient = grad_log_cosh()
        if cost == 4:
            gradient = grad_quantile()
        return gradient