import os

import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dirname = '/mnt/openstack-vm/home/ubuntu/dev/ann-benchmarks/results'
filename = os.path.join(dirname, 'glove-all.txt')

col_names = ['library', 'algo_name', 'build_time', 'best_search_time',
             'best_precision']
df = pd.read_csv(filename, sep='\t', header=None, names=col_names)
filename_hsnw = os.path.join(dirname, 'glove-hnsw.txt')
df_glove = pd.read_csv(filename_hsnw, sep='\t', header=None, names=col_names)

df = pd.concat([df, df_glove])
df['best_queries_per_second'] = 1 / df['best_search_time']
grid = sns.FacetGrid(df, hue='library', legend_out=False)
grid.map(plt.scatter, 'best_precision', 'best_queries_per_second').set(
    yscale='log').add_legend()

all_algos = sorted(set(df.library), key=str.lower)

colors = plt.cm.Set1(np.linspace(0, 1, len(all_algos)))
linestyles = {}
for i, algo in enumerate(all_algos):
    linestyles[algo] = (colors[i], ['--', '-.', '-', ':'][i%4], ['+', '<', 'o', 'D', '*', 'x', 's'][i%7])

handles = []
labels = []

plt.figure()
# plt.figure(figsize=(7, 7))
for algo in all_algos:
    this_df = df[df.library == algo]
    data = this_df.copy()
    data.sort_values(by='best_queries_per_second', ascending=False, inplace=True) # key=lambda t: t[-2]) # sort by time
    # ys = data.best_queries_per_second
    # xs = data.best_precision
    # ls = data.algo_name

    # Plot Pareto frontier
    xs, ys = [], []
    last_y = float('-inf')
    for t in data.itertuples():
        y = t.best_precision
        if y > last_y:
            last_y = y
            xs.append(t.best_precision)
            ys.append(t.best_queries_per_second)
    color, linestyle, marker = linestyles[algo]
    handle, = plt.plot(xs, ys, '-', label=algo, color=color, ms=5, mew=1, lw=2, linestyle=linestyle, marker=marker)
    plt.gca().set_yscale('log')
    plt.gca().set_title('Precision-Performance tradeoff - up and to the right is better')
    plt.gca().set_ylabel('Queries per second ($s^{-1}$) - larger is better')
    plt.gca().set_xlabel('10-NN precision - larger is better')
    box = plt.gca().get_position()
    # plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.gca().legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 9})
    plt.grid(b=True, which='major', color='0.65',linestyle='-')
    plt.xlim([0.0, 1.03])
