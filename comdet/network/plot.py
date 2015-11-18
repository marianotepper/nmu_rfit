import graph_tool as gt
from graph_tool import draw as gt_draw
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn.apionly as sns


def circular_layout(graph, radius=1):
    angle = np.linspace(0, 2 * np.pi, num=graph.num_vertices() + 1)
    pos = graph.new_vertex_property('vector<float>')
    for i, u in enumerate(graph.vertices()):
        pos[u] = [radius * np.cos(angle[i]), radius * np.sin(angle[i])]
    return pos


def create_layout_football(graph):
    value = graph.vertex_properties['value_tsevans']
    independents = [u for u in graph.vertices() if value[u] > 10]
    conferences = set(value[u] for u in graph.vertices() if value[u] <= 10)

    dummy_graph = gt.Graph()
    dummy_graph.add_vertex(n=len(conferences) + len(independents))
    pos_global = circular_layout(dummy_graph)

    pos = graph.new_vertex_property('vector<float>')
    for i, conf in enumerate(conferences):
        vprop_conf = graph.new_vertex_property('bool', vals=False)
        comm = []
        for u in graph.vertices():
            if value[u] == conf:
                vprop_conf[u] = True
                comm.append(u)
        graph.set_vertex_filter(vprop_conf)
        local_pos = circular_layout(graph, 0.15)
        graph.set_vertex_filter(None)
        for u in comm:
            pos[u] = .8 * np.array(local_pos[u]) + pos_global[dummy_graph.vertex(i)]

    vprop_ind = graph.new_vertex_property('bool', vals=False)
    graph.set_vertex_filter(vprop_ind)
    for i, u in enumerate(independents):
        pos[u] = pos_global[dummy_graph.vertex(len(conferences) + i)]
    graph.set_vertex_filter(None)
    return pos


def communities_colormap(n=None):
    if n is None:
        n = 6
    return sns.color_palette('Set1', n)


def communities_node_attributes():
    form_list = range(6)
    return list(itertools.product(form_list, communities_colormap()))


def draw(graph, communities, pos=None, output=None):
    try:
        color = graph.new_vertex_property('vector<float>')
        shape = graph.new_vertex_property('int')
        node_attributes = communities_node_attributes()
        for u in graph.vertices():
            idx = communities[u]
            if idx >= 0:
                color[u] = list(node_attributes[idx][1]) + [0.9]
                shape[u] = node_attributes[idx][0]
            else:
                color[u] = [1, 1, 1, 0.9]
                shape[u] = 0
        gt_draw.graph_draw(graph, pos=pos,
                           vertex_fill_color=color, vertex_shape=shape,
                           edge_color=[0, 0, 0, 0.15],
                           output=output)
    except TypeError:
        augmented_graph = graph.copy()
        edge_communities = []
        for comm in communities:
            restricted = graph.new_vertex_property('bool', vals=False)
            for v in comm:
                restricted[v] = True
            graph.set_vertex_filter(restricted)
            new_edges = list(graph.edges())
            graph.set_vertex_filter(None)
            temp_edges = []
            for e in new_edges:
                e2 = augmented_graph.add_edge(e.source(), e.target())
                temp_edges.append(e2)
            edge_communities.append(temp_edges)

        color_list = communities_colormap(n=len(communities))
        edge_color = augmented_graph.new_edge_property('vector<float>')
        edge_width = augmented_graph.new_edge_property('float', vals=1.0)
        default_colors = np.zeros((4, augmented_graph.num_edges()))
        default_colors[3, :] = 0.15
        edge_color.set_2d_array(default_colors)
        for i, e_comm in enumerate(edge_communities):
            for e in e_comm:
                edge_color[e] = list(color_list[i]) + [0.3]
                edge_width[e] = 10.0
        gt_draw.graph_draw(augmented_graph, pos=pos,
                           vertex_fill_color='white',
                           edge_color=edge_color,
                           edge_pen_width=edge_width,
                           output=output)



def plot_preference_matrix(array, samples, offset=0, labels=None, title=None):
    n_colors = int(np.max(array))
    palette = sns.cubehelix_palette(n_colors + 1, start=2, rot=0, dark=0.15,
                                    light=1)
    cmap = colors.ListedColormap(palette, N=n_colors )
    plt.imshow(array, interpolation='none', cmap=cmap)
    count = 0
    for neigh_comm in samples:
        count += len(neigh_comm) + offset
        plt.plot([count - 0.5] * 2, [-0.5, array.shape[0] - 0.5], 'k')
    plt.tick_params(
        which='both',  # both major and minor ticks are affected
        bottom='off',
        top='off',
        left='off',
        right='off',
        labelbottom='off',
        labelleft='off')
    plt.axis('image')

    cmap = colors.ListedColormap(palette[1:], N=3)
    locs = np.linspace(1, n_colors, n_colors)
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(1, n_colors + 1)
    cb = plt.colorbar(mappable, drawedges=True)
    cb.set_ticks(locs + 0.5)
    if labels is not None:
        cb.set_ticklabels(labels)
    if title is not None:
        cb.ax.set_title(title, loc='left')
    cb.ax.tick_params(left='off', right='off')
