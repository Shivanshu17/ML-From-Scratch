from functools import reduce
import math
def mean_squared_loss(y, pred_y):
    assert len(y) == len(pred_y)
    return (reduce((lambda item1, item2: item1 + item2), map(lambda x: x**2, (y-pred_y)))/len(pred_y))

def mean_absolute_loss(y, pred_y):
    assert len(y) == len(pred_y)
    return (reduce((lambda item1, item2: item1 + item2), map(abs, (y - pred_y)))/len(pred_y))

def huber_loss(y, pred_y, t):
    assert len(y) == len(pred_y)
    summed_up = sum(map(abs, y-pred_y))
    if summed_up < t:
        return (reduce((lambda item1, item2: item1 + item2), map(lambda x:x**2, (y-pred_y)))/len(pred_y))
    else:
        return( t*reduce((lambda item1, item2: item1 + item2), map(abs, (y-pred_y)))/len(pred_y) - (t**2)/2)
    
def log_cosh_loss(y, pred_y):
    assert len(y) == len(pred_y)
    log_cosh = lambda x: math.log(math.cosh(x))
    return (reduce((lambda item1, item2: item1 + item2), map(log_cosh, map(abs, y-pred_y))))

def quantile_loss(y, pred_y, q):
    assert len(y) == len(pred_y)
    assert q<=1 and q>=0
    loss = 0
    for i in range(len(y)):
        if (y[i]-pred_y[i]>0):
            loss = loss + q*(y[i]-pred_y[i])
        else:
            loss = loss + (q-1)*(y[i] - pred_y[i])
    
    return (loss/len(y))


    
