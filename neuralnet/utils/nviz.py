import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend('agg')


def plot(file=None, y=None, query=None, title='', save=False, x_tick_skip=None):
    try:
        df = pd.read_csv(file).query(query) if query else pd.read_csv(file)

        sample_every = max(int(df.shape[0] / 1000), 1)
        df_sample = df.iloc[1::sample_every, :]

        plt.rcParams["figure.figsize"] = [12, 6]
        fig, ax1 = plt.subplots(nrows=1, ncols=1)
        df.plot(y=y, c='#cccccc', ax=ax1, title=title)
        df_sample.plot(y=y, c='#222222', ax=ax1, title=title)

        if x_tick_skip is not None:
            ticks = [int(a) for a in ax1.get_xticks() // x_tick_skip]
            ax1.xaxis.set_ticklabels(ticks)

        ax1.set_xlabel('Epochs')
        if save:
            plt.savefig(file.split('.')[0] + '-' + title + '_' + y + '.png')
        else:
            plt.show()
        plt.close('all')
    except Exception as e:
        print('### PLOTTER ###', e)


def plot_cmap(file=None, query=None, save=False, x=None, y=None, title=''):
    try:
        df = pd.read_csv(file).query(query) if query else pd.read_csv(file)
        plt.rcParams["figure.figsize"] = [12, 8]
        fig, ax1 = plt.subplots(nrows=1, ncols=1)
        z = np.linspace(0, 0.9, df[x].shape[0])
        df.plot.scatter(x=x, y=y, c=z, colormap='magma', ax=ax1, s=40, xlim=(0.5, 1), ylim=(0.5, 1), title=title)
        if save:
            plt.savefig(file.split('.')[0] + '-' + title + '_' + x + '_' + y + '_cmap.png')
        else:
            plt.show()
        plt.close('all')
    except Exception as e:
        print('### PLOTTER ###', e)


def y_scatter(file=None, query=None, y=None, save=False, title='', label=None):
    try:
        df = pd.read_csv(file).query(query) if query else pd.read_csv(file)
        rows = np.arange(df.shape[0])

        plt.rcParams["figure.figsize"] = [8, 8]
        fig, ax1 = plt.subplots(1, 1)
        ax1.scatter(rows, df[y], color='black', s=30)
        ax1.set_title(title)

        if label is not None:
            for i, txt in enumerate(df[label]):
                ax1.annotate(txt[:2], (rows[i], df[y].iloc[i]), xytext=(rows[i] - 0.4, df[y].iloc[i] + 0.006))
        ax1.set_ylabel(y)
        ax1.set_ylim(0.4, 1)
        ax1.grid(True, axis='y')

        if save:
            plt.savefig(file.split('.')[0] + '-' + title + '_' + y + '.png')
        else:
            plt.show()
        plt.close('all')
    except Exception as e:
        print('### PLOTTER ###', e)


def xy_scatter(file=None, query=None, x=None, y=None, label=None, title='', save=False):
    try:
        df = pd.read_csv(file).query(query) if query else pd.read_csv(file)
        plt.rcParams["figure.figsize"] = [8, 8]

        fig, ax1 = plt.subplots(1, 1)
        ax1.scatter(df[x], df[y], color='black', s=30)
        ax1.set_title(title)

        if label is not None:
            for i, txt in enumerate(df[label]):
                ax1.annotate(txt[:2], (df['PRECISION'].iloc[i] + 0.01, df['RECALL'].iloc[i]))
        ax1.set_xlabel('PRECISION')
        ax1.set_ylabel('RECALL')
        ax1.set_xlim((0.4, 1))
        ax1.set_ylim((0.4, 1))
        ax1.grid(True)

        if save:
            plt.savefig(file.split('.')[0] + '-' + title + '_' + x + '_' + y + '.png')
        else:
            plt.show()
        plt.close('all')
    except Exception as e:
        print('### PLOTTER ###', e)

# import neuralnet.mapnet.runs as r
#
# D = r.DRIVE
# file1 = '/home/ak/PycharmProjects/ature/data/DRIVE_MAP/mapnet_logs/THRNET-DRIVE-TEST.csv'
# print(file1)
# xy_scatter(file=file1, title='Test', save=True, x='PRECISION', y='RECALL', label='ID')
