import matplotlib.pyplot as plt
from pathlib import Path

def plot_learning_curve(values_dict: dict, fpath: Path):
    _, ax = plt.subplots(figsize=(8, 15))

    for name, values in values_dict.items():
        ax.plot(values, label = name)
        ax.set_xlabel("Steps")

    ax.set_ylim(0, 2)
    plt.legend()

    plt.savefig(fpath)
    plt.close()
