import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import utils as ut
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
from scipy import interp

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

from pandas.tools.plotting import parallel_coordinates

def add_freq(ax, df):
    ncount = len(df)

    ax2=ax.twinx()

    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()

    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')

    ax2.set_ylabel('Frequency [%]')

    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                ha='center', va='bottom')

    ax2.set_ylim(0,100)
    ax2.grid(None)

def balance_outcome(df):
    ax = sns.countplot(x = df['SeriousDlqin2yrs'] ,palette="Set3")
    sns.set(font_scale=1.5)
    ax.set_ylim(top = 150000)
    ax.set_xlabel(' ')
    ax.set_ylabel(' ')
    fig = plt.gcf()
    fig.set_size_inches(10,5)
    ax.set_ylim(top=160000)

    add_freq(ax, df)

    plt.show()
    
def plot_pca(training_set):
    X = training_set.as_matrix()

    pca = PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    transformed = pd.DataFrame(X)

    plt.scatter(transformed[y==1][0], transformed[y==1][1], label='Class 1', c='red')
    plt.scatter(transformed[y==2][0], transformed[y==2][1], label='Class 2', c='blue')
    plt.scatter(transformed[y==3][0], transformed[y==3][1], label='Class 3', c='lightgreen')

    plt.legend()
    plt.show()