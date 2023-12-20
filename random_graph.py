import networkx as nx
import numpy as np
import os
import json
n_nums = np.arange(20, 500, 5)
p_nums = np.arange(0, 26, 1)/100
datasets = {}
max_degree = 100
for i in n_nums:
    for j in p_nums:
        for k in range(5):
            G = nx.fast_gnp_random_graph(i, j)
            with open('temp.edgelist', 'w') as file:
                file.write(str(i)+' '+str(G.number_of_edges())+'\n')
                for line in nx.generate_edgelist(G, data=False):
                    file.write(line+'\n')
            os.system('cd /home/ubuntu/escape/wrappers && python3 subgraph_counts.py /home/ubuntu/temp.edgelist 4 -i > /dev/null')
            deg_dis = nx.degree_histogram(G)
            if len(deg_dis) < max_degree:
                deg_dis = ' '.join([str(deg_dis[i]) if i < len(deg_dis) else '0' for i in range(max_degree)])
            else:
                deg_dis = ' '.join([str(deg_dis[i]) if i < (max_degree-1) else str(sum(deg_dis[(max_degree-1):])) for i in range(max_degree)])
            with open('/home/ubuntu/escape/wrappers/out.txt') as output:
                counts = [int(float(line.split()[0])) for line in output][2:]
            datasets[deg_dis] = counts

os.system('rm temp.edgelist')
with open('data.json', 'w') as f:
    f.write(json.dumps(datasets, indent=4))