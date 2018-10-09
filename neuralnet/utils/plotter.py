import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

plt.switch_backend('agg')


def plot_csv(file=None, key='LOSS', query=None, title=None, save=False, batches_per_epoch=None):
    try:
        df = pd.read_csv(file).query(query) if query else pd.read_csv(file)

        df_sv = df.copy()
        w = min(21, df.shape[0]-3)
        w = w + 1 if w % 2 == 0 else w
        d = 3 if w > 3 else 1
        df_sv[key] = savgol_filter(df_sv[key], w, d)

        plt.rcParams["figure.figsize"] = [12, 6]
        fig, ax1 = plt.subplots(nrows=1, ncols=1)
        df.plot(y=key, c='#cccccc', ax=ax1, title=title)
        df_sv.plot(y=key, c='#222222', ax=ax1, title=title)

        if batches_per_epoch is not None:
            ticks = [int(a) for a in ax1.get_xticks() // batches_per_epoch]
            ax1.xaxis.set_ticklabels(ticks)

        ax1.set_xlabel('Epochs')
        if save:
            plt.savefig(file.split('.')[0] + '-' + key + '.png')
        else:
            plt.show()
        plt.close('all')
    except Exception:
        traceback.print_exc()


def scattermap_prec_recall(file=None, query=None, save=False):
    try:
        df = pd.read_csv(file).query(query) if query else pd.read_csv(file)
        plt.rcParams["figure.figsize"] = [8, 8]
        fig, ax1 = plt.subplots(nrows=1, ncols=1)
        z = np.linspace(0, 0.9, df['PRECISION'].shape[0])
        df.plot.scatter(x='PRECISION', y='RECALL', c=z, colormap='magma', ax=ax1, s=50, xlim=(0.5, 1), ylim=(0.5, 1))
        if save:
            plt.savefig(file.split('.')[0] + '-PRECLLMAP' + '.png')
        else:
            plt.show()
        plt.close('all')
    except Exception:
        traceback.print_exc()


def scatter_with_id(file=None, query=None, key=None, save=False):
    try:
        df = pd.read_csv(file).query(query) if query else pd.read_csv(file)
        rows = np.arange(df.shape[0])

        plt.rcParams["figure.figsize"] = [8, 8]
        fig, ax1 = plt.subplots(1, 1)
        ax1.scatter(rows, df[key], color='black', s=60)
        for i, txt in enumerate(df['ID']):
            ax1.annotate(txt, (rows[i], df[key].iloc[i]), xytext=(rows[i] - 0.4, df[key].iloc[i] + 0.006))
        ax1.set_ylabel(key)
        ax1.set_ylim(0.4, 1)
        ax1.grid(True, axis='y')

        if save:
            plt.savefig(file.split('.')[0] + '-' + key + '.png')
        else:
            plt.show()
        plt.close('all')
    except Exception:
        traceback.print_exc()


def scatter_prec_recall_with_id(file=None, query=None, save=False):
    try:
        df = pd.read_csv(file).query(query) if query else pd.read_csv(file)
        plt.rcParams["figure.figsize"] = [8, 8]

        fig, ax1 = plt.subplots(1, 1)
        ax1.scatter(df['PRECISION'], df['RECALL'], color='black', s=60)
        for i, txt in enumerate(df['ID']):
            ax1.annotate(txt, (df['PRECISION'].iloc[i] + 0.01, df['RECALL'].iloc[i]))
        ax1.set_xlabel('PRECISION')
        ax1.set_ylabel('RECALL')
        ax1.set_xlim((0.4, 1))
        ax1.set_ylim((0.4, 1))
        ax1.grid(True)

        if save:
            plt.savefig(file.split('.')[0] + '-PRECLL' + '.png')
        else:
            plt.show()
        plt.close('all')
    except Exception:
        traceback.print_exc()
