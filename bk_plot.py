import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from matplotlib.lines import Line2D
import pandas as pd

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), benchmark=None,
                       scoring=None):
    """
    Adapted from here:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """
    _, axes = plt.subplots(1, 1, figsize=(17, 12))
    axes.set_title(title, loc='left', size=20)
    if ylim is not None:
        axes.set_ylim(*ylim)

    axes.set_xlabel("Training examples", size=15)
    axes.set_ylabel("Score", size=15)

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    if benchmark is not None:
        plt.plot(train_sizes[:], np.zeros(len(train_sizes))+benchmark, ls="--",
                 color = "dimgrey", lw=2)
    # Plot learning curve
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="grey")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="green")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    lg = axes.legend(loc="upper right", framealpha=1, fontsize='large')
    
    ax = lg.axes
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='dimgrey', ls='--'))
    labels.append("Benchmark score")
    
    lg._legend_box = None
    lg._init_legend_box(handles, labels)
    lg._set_loc(lg._loc)
    lg.set_title(lg.get_title().get_text())
    return plt, test_scores_mean

def fit_importances(estimator, X, y, title):
    estimator.fit(X, y)
    if hasattr(estimator, 'coef_'):
        
        #get the non-zero coefficients and respective column names
        cols = X.columns[estimator.coef_[0] != 0]
        coefs = estimator.coef_[0][estimator.coef_[0]!=0]
        print("The final linear model has {} predictor variables".format(len(coefs)))
        df = pd.DataFrame(dict(zip((cols), (coefs))), index=['Importance']).T
    else:  
        #get the non-zero coefficients and respective column names
        cols = X.columns
        coefs = estimator.feature_importances_

        df = pd.DataFrame(dict(zip((cols), (coefs))), index=['Importance']).T
    
    y = df.sort_values('Importance', ascending=True)
    axes = y.plot.barh(color='Forestgreen')
    axes.figure.set_size_inches(17,12)
    axes.set_title(title, loc='left', size=20)
    axes.set_ylabel('Feature', size = 15)
    plt.show()
    return plt, df.sort_values('Importance', ascending=False)