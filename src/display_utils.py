import matplotlib.pyplot as plt
from pathlib import Path

def plot_learning_curve(loss_dict: dict, fpath: Path):
    _, ax = plt.subplots(figsize=(14, 8))

    for name, values in loss_dict.items():
        ax.plot(values, label = name)
        ax.set_xlabel("Steps")

    ax.set_ylim(0, 1)
    plt.legend()

    plt.savefig(fpath)
    plt.close()
