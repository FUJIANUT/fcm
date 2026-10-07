from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from .synthetic import ClientData


def pooled_xy(clients: list[ClientData]) -> tuple[np.ndarray, np.ndarray]:
    return np.vstack([client.x for client in clients]), np.concatenate([client.y for client in clients])


def plot_synthetic_comparison(
    clients: list[ClientData],
    centralized_centers: np.ndarray,
    fed_centers: np.ndarray,
    gf_global_centers: np.ndarray,
    gf_personalized_centers: list[np.ndarray],
    relevance: np.ndarray,
    out_path: str | Path,
    title: str,
) -> None:
    x, y = pooled_xy(clients)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Authored at final print size (single-column Elsevier text block is ~6.5 in wide)
    # so that the font sizes set below are also the font sizes on the page.
    with plt.rc_context(
        {
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 7.0,
            "figure.titlesize": 8.5,
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.9), constrained_layout=True)
        axes = axes.ravel()
        colors = plt.get_cmap("tab10")

        for ax in axes[:3]:
            for label in np.unique(y):
                mask = y == label
                ax.scatter(x[mask, 0], x[mask, 1], s=4, alpha=0.35, color=colors(int(label)), label=f"C{label}")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linewidth=0.4, alpha=0.35)

        axes[0].scatter(centralized_centers[:, 0], centralized_centers[:, 1], marker="X", s=60, color="black")
        axes[0].set_title("Centralized FCM")

        axes[1].scatter(fed_centers[:, 0], fed_centers[:, 1], marker="X", s=60, color="black")
        axes[1].set_title("FedFCM")

        axes[2].scatter(
            gf_global_centers[:, 0], gf_global_centers[:, 1],
            marker="s", s=48, facecolor="white", edgecolor="black", linewidth=1.1, label="global",
        )
        for p, centers in enumerate(gf_personalized_centers):
            axes[2].scatter(
                centers[:, 0], centers[:, 1], marker=".", s=18, alpha=0.7,
                color=plt.cm.viridis(p / max(1, len(gf_personalized_centers) - 1)),
            )
        axes[2].set_title("GF-PFedFCM")

        im = axes[3].imshow(relevance, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        axes[3].set_title("Footprint relevance")
        axes[3].set_xlabel("prototype")
        axes[3].set_ylabel("client")
        # Integer prototype/client indices rather than continuous image coordinates.
        axes[3].set_xticks(np.arange(relevance.shape[1]))
        axes[3].set_xticklabels([str(j + 1) for j in range(relevance.shape[1])])
        client_ticks = np.arange(0, relevance.shape[0], max(1, relevance.shape[0] // 6))
        axes[3].set_yticks(client_ticks)
        axes[3].set_yticklabels([str(p + 1) for p in client_ticks])
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles[:4], labels[:4], loc="lower center", ncol=4, frameon=False)
        fig.suptitle(title)
        fig.savefig(out_path, dpi=400)
        plt.close(fig)
