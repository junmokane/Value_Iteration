import numpy as np

n_state = 2
n_action = 2
c, r_A, r_B = 3, 2, 1
a = 0.9
Q_table = np.zeros((n_state, n_action), np.float32)

# initialize transition table
p = np.zeros((n_state, n_action, n_state), np.float32)
p[:, 0, 0] = 1
p[:, 0, 1] = 0
p[:, 1, 0] = 0
p[:, 1, 1] = 1

# initialize cost table
g = np.zeros((n_state, n_action), np.float32)
g[0, 0] = -r_A
g[0, 1] = c - r_B
g[1, 0] = c - r_A
g[1, 1] = -r_B

# run value iteration
for k in range(1000):
    pre_Q_table = np.copy(Q_table)
    for i in range(n_state):
        for u in range(n_action):
            Q_table[i, u] = g[i, u]
            for j in range(n_state):
                Q_table[i, u] += a * p[i, u, j] * np.min(pre_Q_table[j, :])

print(Q_table)
