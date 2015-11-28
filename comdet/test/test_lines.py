from __future__ import absolute_import
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import seaborn.apionly as sns
import scipy.io
import scipy.sparse as sp
import timeit
import comdet.pme as pme
import comdet.pme.acontrario as ac
import comdet.biclustering.preference as pref
import comdet.biclustering.nmf as bc
import comdet.biclustering.deflation as deflation


def base_line_plot(x):
    plt.figure(figsize=(4, 4))
    plt.xlim((-1.1, 1.1))
    plt.ylim((-1.1, 1.1))
    plt.scatter(x[:, 0], x[:, 1], c='w', marker='o')


def plot_models(x, models):
    base_line_plot(x)
    for mod in models:
        mod.plot()


def clean_and_plot(x, original_models, bic_rows, bic_cols, inliers_threshold,
                   epsilon):
    mod_inliers_list = [(pme.Line(x[sp.find(r)[0], :]), r) for r in bic_rows]
    survivors = ac.exclusion_principle(x, mod_inliers_list, inliers_threshold,
                                       epsilon)

    colors = sns.color_palette('Set1', len(bic_rows))

    base_line_plot(x)
    for i, s in enumerate(survivors):
        mod_inliers_list[s][0].plot(color=colors[i], linewidth=3, alpha=0.5)

    base_line_plot(x)
    for i, s in enumerate(survivors):
        for j in sp.find(bic_cols[s])[1]:
            c = mpl_colors.colorConverter.to_rgba(colors[i], alpha=0.2)
            original_models[j].plot(color=c)


def run_sampler(x, n_samples, inliers_threshold, epsilon):
    mdg = pme.model_distance_generator(pme.Line, x, n_samples)
    ig = pme.inliers_generator(mdg, inliers_threshold)

    def meaningful((model, inliers)):
        return ac.meaningful(x, model, inliers, inliers_threshold, epsilon)

    preference_matrix = pref.create_preference_matrix(x.shape[0])
    original_models = []
    for i, (model, inliers) in enumerate(ac.filter_meaningful(meaningful, ig)):
        preference_matrix = pref.add_col(preference_matrix, inliers)
        original_models.append(model)

    plot_models(x, original_models)

    print 'Preference matrix size:', preference_matrix.shape

    plt.figure()
    pref.plot_preference_matrix(preference_matrix)

    print 'Running regular bi-clustering'
    t = timeit.default_timer()
    online_deflator = deflation.Deflator(preference_matrix)
    rows, cols = bc.bicluster(online_deflator)
    t1 = timeit.default_timer() - t
    print 'Time:', t1

    clean_and_plot(x, original_models, rows, cols, inliers_threshold, epsilon)

    print 'Running compressed bi-clustering'
    t = timeit.default_timer()
    compression_level = 64
    online_deflator = deflation.L1CompressedDeflator(preference_matrix,
                                                     compression_level)
    t1 = timeit.default_timer() - t
    print 'Initialization time:', t1

    t = timeit.default_timer()
    rows, cols = bc.bicluster(online_deflator)
    t2 = timeit.default_timer() - t
    print 'Time:', t2

    clean_and_plot(x, original_models, rows, cols, inliers_threshold, epsilon)


def test():
    # np.random.seed(0)
    examples = scipy.io.loadmat('../data/JLinkageExamples.mat')
    star_examples = filter(lambda s: s.find('Star') == 0, examples.keys())

    name = star_examples[3]
    data = examples[name].T
    print name, data.shape

    n_samples = 5000
    inliers_threshold = 0.03
    epsilon = 1
    run_sampler(data, n_samples, inliers_threshold, epsilon)


if __name__ == '__main__':
    # plt.switch_backend('TkAgg')
    test()
    plt.show()