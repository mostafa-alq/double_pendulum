# Test neural network setup
arr = [[0, 0, 1],[1, 0, 3], [0, 1, 4], [2, 1, 8],[-1, 2, 5]]

w = [0, 0]
b = 0
lr = 0.01

def predict(x, w, b):
    total = 0
    for xi, wi in zip(x, w):
        total += xi * wi
    total += b
    return total

def loss():
    err_arr = []
    for i in range(0, len(arr)):
        y_predict = predict((arr[i][0], arr[i][1]), w, b)
        err_arr.append((arr[i][2] - y_predict) ** 2)

    return sum(err_arr) / len(err_arr)

def derivatives():
    deltaw = []
    deltab = []
    for i in range(0, len(arr)):
        deltaw.append(2 * (predict((arr[i][0], arr[i][1]), w, b) - arr[i][1]) * arr[i][0])
        deltab.append(2 * (predict((arr[i][0], arr[i][1]), w, b) - arr[i][1]))
    return sum(deltaw) / len(deltaw), sum(deltab) / len(deltab)

for i in range(0,500):
    dw, db = derivatives()
    w = w - lr * dw
    b = b - lr * db
    if i % 50 == 0:
        print(w, b)
        print(loss())
        print(derivatives())