
# coding: utf-8

# In[1]:

# AIによる着手、評価値の出力

import random as rd
import numpy as np
import matplotlib.pyplot as plt
from gomoku import *
from value_network import *
from mymath import *


# In[2]:


# AIによる着手のアルゴリズム
def ai_put(g, model, random=0.):
    if rd.random() < random:
        g.rand_put()
        return g
    next_nodes = g.next_nodes()
    next_values = []
    for node in next_nodes:
        if node.end_game() == g.turn:
            return node
        value = (1.0 if g.turn == 1 else -1.0) * model.out([node.square])[0][0]
        next_values.append(value)
    next_values = np.array(next_values)
    if random == 0:
        g = next_nodes[np.argmax(next_values)]
    else:
        next_values = softmax(next_values, random)
        index = np.argmax(np.random.multinomial(1, next_values))
        g = next_nodes[index]
    return g


# In[ ]:


# AIによる評価値色出力のアルゴリズム
def value_out(g, model):
    out_map = [[[0, 0, 0] for _ in range(g.size)] for __ in range(g.size)]
    max_v = -float('inf')
    min_v = float('inf')
    for i in range(g.size):
        for j in range(g.size):
            s = g.square[i][j]
            v = 0
            if s == [0, 0]:
                new_g = copy.deepcopy(g)
                new_g.put(i, j)
                v = (1.0 if g.turn == 1 else -1.0) * model.out([new_g.square])[0]
                max_v = max(max_v, v)
                min_v = min(min_v, v)
            v = int((v * 255 + 255) / 2)
            if s == [1, 0]:
                out_map[i][j] = [0, 0, 0]
            elif s == [0, 1]:
                out_map[i][j] = [255, 255, 255]
            else:
                if v > 127:
                    out_map[i][j] = [v, 255-v, 0]
                else:
                    out_map[i][j] = [0, v, 255-v]
#     print('MAX: ', max_v, ' | MIN: ' min_v)
    plt.imshow(np.array(out_map))
    plt.title("Value Image")
    plt.show()

