import glob
import cPickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

if __name__ == '__main__':

    # Specify result files
    path = '*.pkl'
    files = glob.glob(path)

    for result in files:
        with open(result, 'rb') as f:
            data = cPickle.load(f)

            [train_data, valid_data, test_data, gradient_data, duration, weights] = data[0]
            f.close()

            print(\
                'Data file %s \n'
                'Last training error %f \n'
                'Last validation error %f\n'
                'Best testing error %f \n'
                'Duration %d'
            ) % (result, train_data[-1]*100, valid_data[-1]*100, test_data[-1]*100, duration/60)

            weight = []

            for w_index in xrange(len(weights)-1):
                if w_index % 2 == 0:
                    weight.extend(weights[w_index].T)

            weight = np.asarray(weight)

            outer = gridspec.GridSpec(10, 10, wspace=0, hspace=0)
            fig = plt.figure(figsize=(10, 10))

            weight = weight[0:100]
            # use global min / max to ensure all weights are shown on the same scale
            vmin, vmax = weight.min(), weight.max()

            for j in xrange(100):
                ax = plt.Subplot(fig, outer[j])
                ax.matshow(weight[j].reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin, vmax=.5 * vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)

    plt.savefig('NC.eps', format='eps', bbox_inches='tight')

