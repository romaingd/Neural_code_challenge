import numpy as np
import matplotlib.pyplot as plt

def plot_avg_feature_importance(importances, feature_names):
    if len(importances.shape) == 1:
        importances = importances.reshape(1, -1)
    mean_importances = np.mean(importances, axis=0)
    e = np.argsort(mean_importances)[::-1][:40]
    std_importances = np.std(importances, axis=0)
    fig, ax = plt.subplots(figsize=(13,13))
    ax.barh(np.arange(len([feature_names[i] for i in e])), mean_importances[e],
            xerr=std_importances[e], align='center')
    ax.set_yticks(np.arange(len([feature_names[i] for i in e])))
    ax.set_yticklabels([feature_names[i] for i in e])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Feature importance and variability')
    plt.tight_layout()
    plt.show()