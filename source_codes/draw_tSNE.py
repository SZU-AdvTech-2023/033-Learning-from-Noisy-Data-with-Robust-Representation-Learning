import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import manifold
import numpy as np
import seaborn as sns


def visual(feat):
    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)

    x_ts = ts.fit_transform(feat)

    print(x_ts.shape)  # [num, 2]

    x_min, x_max = x_ts.min(0), x_ts.max(0)

    x_final = (x_ts - x_min) / (x_max - x_min)

    return x_final


# prefix_path = 'visualization/openset_svhn/'
prefix_path = 'visualization/cifar10_asym_0.4_ramp/'
# prefix_path = 'visualization/openset_cifar100/'
checkpoints = ['10', '50', '100', '150', '200']

for checkpoint in checkpoints:
    # load npz file
    checkpoint = checkpoint
    data = np.load(os.path.join(prefix_path, f'features_{checkpoint}.npz'))
    features = data['features']
    labels = data['labels']
    clean_labels = data['clean_labels']

    reduced_features = visual(features)
    df = pd.DataFrame(reduced_features, columns=['tsne-2d-one', 'tsne-2d-two'])
    # cols = ['#C0C0C0', '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']
    cols = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']
    df['y'] = clean_labels
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=cols,
        data=df,
        legend="full",
        alpha=0.5,
    )
    # remove axis
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig(os.path.join(prefix_path, f'tsne2D_{checkpoint}.png'), bbox_inches='tight', dpi=1200)