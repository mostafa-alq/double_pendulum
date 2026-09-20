# Test neural network setup - x1, x2, y1, y2
arr = [[0, 0, 1, 0], [1, 0, 3, 1], [0, 1, 4, -1], [2, 1, 8, 1], [-1, 2, 5, -3]]

w1 = [[0.5, -0.4], [0.3, 0.8], [-0.6, 0.2]]
b1 = [0.1, 0.0, -0.1]
w = [[0.2, -0.5, 0.4], [0.7, 0.1, -0.3]]
b = [0.0, 0.0]
lr = 0.01

# Dot product + bias per layer
def predict(x, w, b):
    result = []
    for w_neuron, b_neuron in zip(w, b):
        total = 0
        for xi, wi in zip(x, w_neuron):
            total += xi * wi
        total += b_neuron
        result.append(total)
    return result

# RELU replaces all negative numbers with 0
def relu(values):
    out = []
    for v in values:
        out.append(max(0, v))
    return out

# Forward pass
def forward(x):
    hidden = relu(predict(x, w1, b1))
    return predict(hidden, w, b)

# Sum of MSE across all rows
def loss():
    err_arr = []
    for i in range(0, len(arr)):
        y_predict1, y_predict2 = forward((arr[i][0], arr[i][1]))
        err_arr.append((y_predict1 - arr[i][2])**2 + (y_predict2 - arr[i][3])**2)

    return sum(err_arr) / len(err_arr)

# Calculate the loss per nudge
def derivatives():
    n_rows = len(arr)
    dw = []
    for w_neuron in w:
        dw.append([0] * len(w_neuron))
    db = [0] * len(b)

    for row in arr:
        x = (row[0], row[1])
        targets = (row[2], row[3])
        hidden = relu(predict(x, w1, b1))
        outputs = predict(hidden, w, b)
        for n in range(len(w)):
            error = outputs[n] - targets[n]
            for j in range(len(hidden)):
                dw[n][j] += 2 * error * hidden[j] / n_rows
            db[n] += 2 * error / n_rows
    return dw, db


# Measure the same gradients by nudging each parameter and measuring the loss
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

# Prints the largest gap between derivatives() and the measured gradients (should be ~1e-10)
def check_gradients():
    num_dw, num_db = numerical_gradients()
    dw, db = derivatives()
    worst = 0
    for n in range(len(w)):
        for j in range(len(w[n])):
            worst = max(worst, abs(num_dw[n][j] - dw[n][j]))
        worst = max(worst, abs(num_db[n] - db[n]))
    print(worst)

check_gradients()

for i in range(0,500):
    dw, db = derivatives()

    new_w = []
    for w_neuron, dw_neuron in zip(w, dw):
        new_w_neuron = []
        for wi, dwi in zip(w_neuron, dw_neuron):
            new_w_neuron.append(wi - lr * dwi)
        new_w.append(new_w_neuron)
    w = new_w

    new_b = []
    for bi, dbi in zip(b, db):
        new_b.append(bi - lr * dbi)
    b = new_b

    if i % 50 == 0:
        print(w, b)
        print(loss())
        print(derivatives())

print(forward((1, 2)))
