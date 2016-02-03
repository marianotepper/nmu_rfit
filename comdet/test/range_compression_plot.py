import matplotlib.pyplot as plt
import seaborn

compressions = [8, 16, 32, 64, 128]


def plot(vals_mean, vals_std, vals_median, title, ylabel=None):
    plt.figure()
    plt.bar(range(len(compressions)), vals_mean, yerr=vals_std, align='center',
            ecolor='k')
    for i in range(len(compressions)):
        plt.plot([i - .4, i + .4], [vals_median[i]] * 2, '#e41a1c')
    plt.xticks(range(len(compressions)), compressions)
    plt.xlabel('Compression level')
    if ylabel is not None:
        plt.ylabel(ylabel.capitalize())
    plt.title(title.capitalize())
    plt.savefig('range_compression_' + title + '.pdf', dpi=600)


time_mean = [2.352, 2.658, 2.549, 2.423, 2.814]
time_std = [0.484, 0.553, 0.510, 0.489, 0.447]
time_median = [2.420, 2.661, 2.671, 2.601, 2.843]
plot(time_mean, time_std, time_median, 'Time', 'seconds')

gnmi_mean = [0.604, 0.584, 0.618, 0.617, 0.591]
gnmi_std = [0.097, 0.072, 0.105, 0.101, 0.114]
gnmi_median = [0.620, 0.609, 0.617, 0.622, 0.598]
plot(gnmi_mean, gnmi_std, gnmi_median, 'gnmi')

prec_mean = [0.764, 0.743, 0.775, 0.780, 0.753]
prec_std = [0.108, 0.094, 0.106, 0.106, 0.114]
prec_median = [0.791, 0.765, 0.804, 0.831, 0.778]
plot(prec_mean, prec_std, prec_median, 'precision')

rec_mean = [0.912, 0.921, 0.919, 0.920, 0.908]
rec_std = [0.064, 0.051, 0.072, 0.070, 0.069]
rec_median = [0.938, 0.938, 0.939, 0.949, 0.939]
plot(rec_mean, rec_std, rec_median, 'recall')

plt.show()
