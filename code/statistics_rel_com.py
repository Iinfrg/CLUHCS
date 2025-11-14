import numpy as np
from collections import Counter

data_name = 'self_imdb'
path = './embeds/' + data_name + "/labels.npy"
labels = np.load(path)
comm_counter = Counter(labels.tolist())

for comm_id, size in comm_counter.items():
    print(f"社区 {comm_id}: 大小 = {size}")
