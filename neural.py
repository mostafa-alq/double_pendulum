# Test neural network setup - x1, x2, y1, y2
arr = [[0, 0, 1, 0], [1, 0, 3, 1], [0, 1, 4, -1], [2, 1, 8, 1], [-1, 2, 5, -3]]

w1 = [[0.5, -0.4], [0.3, 0.8], [-0.6, 0.2]]
b1 = [0.1, 0.0, -0.1]
w = [[0.2, -0.5, 0.4], [0.7, 0.1, -0.3]]
b = [0.0, 0.0]
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

def relu(values):
    out = []
    for v in values:
        out.append(max(0, v))
    return out

def forward(x):
    hidden = relu(predict(x, w1, b1))
    return predict(hidden, w, b)

def loss():
    err_arr = []
    for i in range(0, len(arr)):
        y_predict1, y_predict2 = forward((arr[i][0], arr[i][1]))
        err_arr.append((y_predict1 - arr[i][2])**2 + (y_predict2 - arr[i][3])**2)

    return sum(err_arr) / len(err_arr)

def derivatives():
    deltaw1_1, deltaw1_2, deltab1 = [], [], []
    deltaw2_1, deltaw2_2, deltab2 = [], [], []
    
    for i in range(0, len(arr)):
        y_predict1, y_predict2 = predict((arr[i][0], arr[i][1]), w, b)
        error1 = y_predict1 - arr[i][2]
        error2 = y_predict2 - arr[i][3]

        deltaw1_1.append(2 * error1 * arr[i][0])
        deltaw1_2.append(2 * error1 * arr[i][1])
        deltab1.append(2 * error1)

        deltaw2_1.append(2 * error2 * arr[i][0])
        deltaw2_2.append(2 * error2 * arr[i][1])
        deltab2.append(2 * error2)

    avg_deltaw1_1 = sum(deltaw1_1) / len(deltaw1_1)
    avg_deltaw1_2 = sum(deltaw1_2) / len(deltaw1_2)
    avg_deltab1 = sum(deltab1) / len(deltab1)

    avg_deltaw2_1 = sum(deltaw2_1) / len(deltaw2_1)
    avg_deltaw2_2 = sum(deltaw2_2) / len(deltaw2_2)
    avg_deltab2 = sum(deltab2) / len(deltab2)

    return [[avg_deltaw1_1, avg_deltaw1_2], [avg_deltaw2_1, avg_deltaw2_2]], [avg_deltab1, avg_deltab2]

def numerical_gradients(eps=1e-5):
    num_dw = []
    # Per neuron
    for n in range(len(w)):
        row = []
        # Per neuron weight
        for j in range(len(w[n])):
            original = w[n][j]
            w[n][j] = original + eps
            loss_plus = loss()
            w[n][j] = original - eps
            loss_minus = loss()
            w[n][j] = original
            row.append((loss_plus - loss_minus) / (2 * eps))
        num_dw.append(row)

    num_db = []
    # Per neuron bias
    for n in range(len(b)):
        original = b[n]
        b[n] = original + eps
        loss_plus = loss()
        b[n] = original - eps
        loss_minus = loss()
        b[n] = original
        num_db.append((loss_plus - loss_minus) / (2 * eps))

    return num_dw, num_db

def check_gradients():
    num_dw, num_db = numerical_gradients()
    dw, db = derivatives()
    worst = 0
    for n in range(len(w)):
        for j in range(len(w[n])):
            worst = max(worst, abs(num_dw[n][j] - dw[n][j]))
        worst = max(worst, abs(num_db[n] - db[n]))
    print(worst)

# check_gradients()

# for i in range(0,500):
#     dw, db = derivatives()

#     new_w = []
#     for w_neuron, dw_neuron in zip(w, dw):
#         new_w_neuron = []
#         for wi, dwi in zip(w_neuron, dw_neuron):
#             new_w_neuron.append(wi - lr * dwi)
#         new_w.append(new_w_neuron)
#     w = new_w

#     new_b = []
#     for bi, dbi in zip(b, db):
#         new_b.append(bi - lr * dbi)
#     b = new_b

#     if i % 50 == 0:
#         print(w, b)
#         print(loss())
#         print(derivatives())

print(forward((1, 2)))
