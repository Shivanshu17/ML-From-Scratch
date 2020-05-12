from functools import reduce
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
    

