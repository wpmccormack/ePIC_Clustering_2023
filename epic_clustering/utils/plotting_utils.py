import numpy as np
import matplotlib.pyplot as plt

def plot_clusters(layer, x_bounds=None, y_bounds=None):
    # make color map with as many entries as labels
    cmap = plt.cm.get_cmap('viridis', len(layer.dbscan_label.unique()))
    # plot each cluster
    for i, label in enumerate(layer.dbscan_label.unique()):
        if label == -1:
            continue
        plt.scatter(layer.posx[layer.dbscan_label == label], layer.posy[layer.dbscan_label == label], label=f'Cluster{label}', alpha=0.5, color=cmap(i))
        cluster_posx, cluster_posy, cluster_posz, cluster_posx_std, cluster_posy_std, cluster_posz_std = get_cluster_pos(layer, label)
        plt.scatter(cluster_posx, cluster_posy, label=f'Cluster{label} pos', alpha=0.5, s=200, marker='x', c=cmap(i))
        circle = plt.Circle((cluster_posx, cluster_posy), 3*cluster_posx_std, fill=False, color=cmap(i))
        plt.gcf().gca().add_artist(circle)        

    plt.scatter(layer.posx[layer.dbscan_label == -1], layer.posy[layer.dbscan_label == -1], label='Noise', color='black', alpha=0.2)
    if x_bounds is not None:
        plt.xlim(x_bounds)
    if y_bounds is not None:
        plt.ylim(y_bounds)
    plt.legend()

# cluster posx is mean of cluster posx entries weighted by energy
def get_cluster_pos(layer, cluster):
    cluster_posx = np.average(layer.posx[layer.dbscan_label == cluster], weights=layer[layer.dbscan_label == cluster].E)
    cluster_posy = np.average(layer.posy[layer.dbscan_label == cluster], weights=layer[layer.dbscan_label == cluster].E)
    cluster_posz = np.average(layer.posz[layer.dbscan_label == cluster], weights=layer[layer.dbscan_label == cluster].E)
    cluster_posx_std = np.sqrt(np.average((layer.posx[layer.dbscan_label == cluster] - cluster_posx)**2, weights=layer[layer.dbscan_label == cluster].E))
    cluster_posy_std = np.sqrt(np.average((layer.posy[layer.dbscan_label == cluster] - cluster_posy)**2, weights=layer[layer.dbscan_label == cluster].E))
    cluster_posz_std = np.sqrt(np.average((layer.posz[layer.dbscan_label == cluster] - cluster_posz)**2, weights=layer[layer.dbscan_label == cluster].E))
    return cluster_posx, cluster_posy, cluster_posz, cluster_posx_std, cluster_posy_std, cluster_posz_std