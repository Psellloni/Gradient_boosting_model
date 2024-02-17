# MSE
def mean_squared_error(y_true, y_pred):
    if len(y_true) != len(y_pred):
        print('size of arguments should be equal')
        return
    
    else:
        result = 0

        for i in range(len(y_true)):
            result += (y_true[i] - y_pred[i]) ** 2
        
        return (result) / len(y_true)
    

# MAE
def mean_absolute_error(y_true, y_pred):
    if len(y_true) != len(y_pred):
        print('size of arguments should be equal')
        return
    
    else:
        result = 0

        for i in range(len(y_true)):
            result += abs(y_true[i] - y_pred[i])
        
        return (result) / len(y_true)