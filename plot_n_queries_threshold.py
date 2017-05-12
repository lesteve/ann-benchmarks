import os
import argparse

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--output')

args = parser.parse_args()

# dirname = '/mnt/openstack-vm/dev/ann-benchmarks/results'
# filename = os.path.join(dirname, 'sift-10000.txt')

filename = args.input

col_names = ['library', 'algo_name', 'build_time', 'best_search_time',
             'best_precision']
df = pd.read_csv(filename, sep='\t', header=None, names=col_names)
df['best_queries_per_second'] = 1 / df['best_search_time']

# Fixes up build_time for hnsw (because of caching only one mode per M
# is actually trained, the other reuse the index that has already been
# computed)
hnsw_mask = df.algo_name.str.contains('hnsw')
hnsw_df = df[hnsw_mask]
hnsw_build_type = hnsw_df.algo_name.str.replace(
    '.+index_param=..(M=\d+).+', r'\1')
hnsw_df = df[hnsw_mask]
build_time = hnsw_df.groupby(hnsw_build_type)['build_time'].max()
df.loc[hnsw_mask, 'build_time'] = build_time[hnsw_build_type].values

brute_force_blas_series = df[
    df.library.str.contains('bruteforce-blas')].iloc[0]
build_time_ref = brute_force_blas_series.build_time
query_time_ref = brute_force_blas_series.best_search_time

# This is the number of queries for which the shorter query time make
# up for the time spent in fit (the reference is brute-force-blas)
df['n_queries_threshold'] = ((df.build_time - build_time_ref) /
                             (query_time_ref - df.best_search_time))

annoy_mask = df.library.str.contains('annoy')
# to_plot_df = df[df.n_queries_threshold > 0 & (hnsw_mask | annoy_mask)]

all_algos = ['annoy', 'ball', 'bruteforce', 'bruteforce-blas',
             'hnsw(nmslib)', 'kd', 'lshf']

colors = plt.cm.Set1(np.linspace(0, 1, len(all_algos)))
linestyles = {}
for i, algo in enumerate(all_algos):
    linestyles[algo] = (colors[i], ['+', '<', 'o', 'D', '*', 'x', 's'][i % 7])

label = 'hnsw(nmslib)'
ax = df[hnsw_mask & (df.n_queries_threshold > 0)].plot(
    x='best_precision', y='n_queries_threshold', kind='scatter',
    c=linestyles[label][0], marker=linestyles[label][1],
    label=label)

label = 'annoy'
df[annoy_mask & (df.n_queries_threshold > 0)].plot(
    x='best_precision', y='n_queries_threshold', kind='scatter',
    c=linestyles[label][0], marker=linestyles[label][1],
    label=label, ax=ax)
plt.gca().set_yscale('log')
plt.ylim([1e3, 1e6])
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.savefig(args.output, bbox_inches='tight')
