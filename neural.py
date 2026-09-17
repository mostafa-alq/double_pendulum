# Test neural network setup - x1, x2, y1, y2
arr = [[0, 0, 1, 0], [1, 0, 3, 1], [0, 1, 4, -1], [2, 1, 8, 1], [-1, 2, 5, -3]]

w = [[0, 0], [0, 0]]
b = [0, 0]
lr = 0.01

def predict(x, w, b):
    result = []
    for w_neuron, b_neuron in zip(w, b):
        total = 0
        for xi, wi in zip(x, w_neuron):
            total += xi * wi
        total += b_neuron
        result.append(total)
    return result

def loss():
    err_arr = []
    for i in range(0, len(arr)):
        y_predict = predict((arr[i][0], arr[i][1]), w, b)
        err_arr.append((arr[i][2] - y_predict) ** 2)

    return sum(err_arr) / len(err_arr)

def derivatives():
    deltaw1 = []
    deltaw2 = []
    deltab = []
    for i in range(0, len(arr)):
        deltaw1.append(2 * (predict((arr[i][0], arr[i][1]), w, b) - arr[i][2]) * arr[i][0])
        deltaw2.append(2 * (predict((arr[i][0], arr[i][1]), w, b) - arr[i][2]) * arr[i][1])
        deltab.append(2 * (predict((arr[i][0], arr[i][1]), w, b) - arr[i][2]))
    return (sum(deltaw1) / len(deltaw1), sum(deltaw2) / len(deltaw2)), sum(deltab) / len(deltab)

for i in range(0,500):
    dw, db = derivatives()
    new_w = []
    for wi, dwi in zip(w, dw):
        new_w.append(wi - lr * dwi)
    w = new_w
    b = b - lr * db
    if i % 50 == 0:
        print(w, b)
        print(loss())
        print(derivatives())