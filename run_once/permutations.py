import numpy as np

test = []

m = 3
c = [0, 1, 2, 3]
k = len(c)
p = [0,0.25, 0.5, 0.75, 1.0]

for j in range(100000):
    v = np.random.choice(c, m, replace=True)
    #print(v)
    # turn v into 1-hot matrix
    v_ = np.zeros((m, k))
    #print(v_)
    v_[np.arange(m), v] = 1
    #print(v_)
    p_ = np.random.choice(p, m, replace=True)
    #print(p_)
    p_ = np.repeat(p_, k).reshape(m, k)
    #print(p_)
    v_ = v_ * p_
    #print(v_)
    v_ = v_.sum(axis=0)
    #print(v_)
    n = np.unique(v).size
    if n == 1:
        x = np.sum(v_)
    else:
        x = np.max(v_) - np.sum(v_) / (n - 1) + np.max(v_) / (n - 1)
    #print(x)
    x /= m
    #print(x)
    if np.isnan(x):
        continue
    if x > 0 and x < 0.0001:
        continue
    test.append(x)


print(np.unique(np.round(test, 10)).size)
