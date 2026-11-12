import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
out=Path(__file__).resolve().parents[1]
cl=json.load(open(out/'cluster_details.json'))
# PCA is only a display, not the clustering input.
fig,axes=plt.subplots(2,2,figsize=(12,9))
for ax,c in zip(axes.flat,'REGQ'):
 z=np.load(out/f'embeddings/{c}.npz');x=z['vectors'];ids=z['ids'].tolist();labels={s:g['cluster'] for g in cl[c] for s in g['members']};pca=PCA(n_components=2,random_state=42);xy=pca.fit_transform(x);ax.scatter(xy[:,0],xy[:,1],c=[labels[s] for s in ids],s=14,cmap='tab10',alpha=.65);ax.set_title(f'{c}: n={len(ids)}, PCA variance={pca.explained_variance_ratio_.sum():.1%}');ax.set_xlabel('Principal component 1');ax.set_ylabel('Principal component 2')
fig.suptitle('Exploratory clusters: K-means k=10 on original 1024-D vectors\n2-D PCA is visualization only; colors are not verified error types');fig.tight_layout();fig.savefig(out/'cluster_pca.png',dpi=160);plt.close(fig)
