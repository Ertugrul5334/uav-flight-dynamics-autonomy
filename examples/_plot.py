"""Optional plotting helper: figures are saved only if matplotlib is installed."""

from pathlib import Path

OUT = Path(__file__).resolve().parent / "output"


def save_figure(name, panels):
    """``panels``: list of (title, x, [(y, label), ...], ylabel)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping plot)")
        return None
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 2.4 * len(panels)), sharex=True)
    axes = axes if len(panels) > 1 else [axes]
    for ax, (title, x, series, ylabel) in zip(axes, panels):
        for y, label in series:
            ax.plot(x, y, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if len(series) > 1:
            ax.legend(fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"saved {path}")
    return path
