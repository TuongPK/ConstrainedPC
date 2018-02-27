import glob
import cPickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = '*.pkl'

    files = glob.glob(path)

    # Labels
    name = []

    count = -1

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    for result in files:
        print result
        count += 1

        with open(result, 'rb') as f:
            data = cPickle.load(f)

            [train_data, valid_data, test_data, gradient_data, duration, weights] = data[0]
            f.close()

        ax1.plot(train_data, label=name[count])
        ax2.plot(valid_data, label=name[count])

    ax2.legend()
    ax1.set_xlabel('Epochs')
    ax2.set_xlabel('Epochs')
    ax1.set_ylabel('Training Error')
    ax2.set_ylabel('Validation Error')

    fig.subplots_adjust(hspace=0)
    plt.savefig('error_rate.eps', format='eps', dpi=1000, bbox_inches='tight')
