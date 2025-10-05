import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y, title=None, savepath=None, grid_step=0.02):
    if X.shape[1] != 2:
        raise ValueError("visualize_classifier очікує рівно 2 ознаки (два стовпці у X).")

    # Межі поля
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    # Прогноз на сітці
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.35)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    plt.xlabel('x1')
    plt.ylabel('x2')
    if title is None:
        title = f"Decision regions: {classifier.__class__.__name__}"
    plt.title(title)
    plt.tight_layout()

    if savepath is None:
        import os, time
        os.makedirs("outputs", exist_ok=True)
        fname = f"outputs/vis_{classifier.__class__.__name__}_{int(time.time())}.png"
    else:
        fname = savepath

    plt.savefig(fname, dpi=150)
    plt.close()
    return fname
