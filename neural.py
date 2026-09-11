# Test neural network setup
arr = [[0, 1],[1, 3], [2,5], [3,7],[4,9]]

w = 0
b = 0

def predict(x, w, b):
    return w * x + b

def loss():
    err_arr = []
    for i in range(0, len(arr)):
        y_predict = predict(arr[i][0], w, b)
        err_arr.append((arr[i][1] - y_predict) ** 2)

    mse = sum(err_arr) / len(err_arr)
    return mse

print(loss())