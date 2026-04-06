"""
SC2 Dataset Inspector
=====================
Visualise le dataset produit par sc2_dataset_builder.py

Usage :
    python inspect_dataset.py --dataset ./dataset
    python inspect_dataset.py --dataset ./dataset --heatmap heatmap_t600s_p1.npy --idx 0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_dataset(dataset_dir: Path):
    csv_path = dataset_dir / "dataset.csv"
    meta_path = dataset_dir / "metadata.json"

    df = pd.read_csv(csv_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return df, meta


def print_summary(df: pd.DataFrame, meta: dict):
    print("\n══════════════════════════════════════════")
    print("  SC2 Dataset — Résumé")
    print("══════════════════════════════════════════")
    print(f"  Snapshots totaux : {len(df)}")
    print(f"  Replays          : {meta['total_replays']} ({meta['errors']} erreurs)")
    print(f"  Timestamps (s)   : {meta['timestamps_s']}")
    print(f"  Grille heatmap   : {meta['grid_size']}×{meta['grid_size']}")
    print(f"\n  Distribution des labels :")
    lmap = meta["label_mapping"]
    for k, v in df["label"].value_counts().sort_index().items():
        print(f"    {lmap[str(k)]:20s} : {v:5d} ({100*v/len(df):.1f}%)")
    print(f"\n  Colonnes : {list(df.columns)}")
    print(f"\n  Exemple (première ligne) :")
    print(df.iloc[0].to_string())
    print("══════════════════════════════════════════\n")


def plot_eco_distributions(df: pd.DataFrame, output_dir: Path):
    features = ["p1_minerals", "p1_army_count", "p1_workers_count",
                "delta_minerals", "delta_army", "delta_supply"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Distribution des features économiques", fontsize=14)

    for ax, feat in zip(axes.flat, features):
        for label, color in [(1, "#4CAF50"), (2, "#F44336"), (0, "#9E9E9E")]:
            subset = df[df["label"] == label][feat]
            ax.hist(subset, bins=40, alpha=0.5, color=color,
                    label=f"label={label}", density=True)
        ax.set_title(feat)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / "eco_distributions.png"
    plt.savefig(out, dpi=150)
    print(f"  Graphique sauvegardé : {out}")
    plt.close()


def plot_heatmap_sample(heatmap_path: Path, idx: int, output_dir: Path):
    arr = np.load(heatmap_path)  # (N, H, W)
    if idx >= len(arr):
        print(f"  Index {idx} hors limites (max {len(arr)-1})")
        return

    hmap = arr[idx]
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(hmap, cmap="hot", origin="lower", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"{heatmap_path.name} — replay #{idx}")
    ax.set_xlabel("X (normalisé)")
    ax.set_ylabel("Y (normalisé)")

    out = output_dir / f"heatmap_sample_{heatmap_path.stem}_idx{idx}.png"
    plt.savefig(out, dpi=150)
    print(f"  Heatmap sauvegardée : {out}")
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path):
    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in ("label", "timestamp_s", "p1_player_id", "p2_player_id")]
    corr = df[numeric_cols + ["label"]].corr()["label"].drop("label").sort_values()

    fig, ax = plt.subplots(figsize=(8, max(4, len(corr) * 0.25)))
    colors = ["#F44336" if v < 0 else "#4CAF50" for v in corr]
    corr.plot(kind="barh", ax=ax, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Corrélation des features avec le label")
    ax.set_xlabel("Corrélation de Pearson")
    plt.tight_layout()

    out = output_dir / "feature_correlations.png"
    plt.savefig(out, dpi=150)
    print(f"  Corrélations sauvegardées : {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="SC2 Dataset Inspector")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--heatmap", type=str, default=None,
                        help="Nom du fichier .npy à visualiser")
    parser.add_argument("--idx", type=int, default=0,
                        help="Index du replay dans le .npy (défaut: 0)")
    args = parser.parse_args()

    df, meta = load_dataset(args.dataset)
    print_summary(df, meta)

    print("  Génération des graphiques...")
    plot_eco_distributions(df, args.dataset)
    plot_correlation_matrix(df, args.dataset)

    if args.heatmap:
        hmap_path = args.dataset / args.heatmap
        if hmap_path.exists():
            plot_heatmap_sample(hmap_path, args.idx, args.dataset)
        else:
            print(f"  Fichier introuvable : {hmap_path}")
    else:
        # Auto : première heatmap disponible
        npy_files = list(args.dataset.glob("heatmap_*.npy"))
        if npy_files:
            plot_heatmap_sample(npy_files[0], 0, args.dataset)


if __name__ == "__main__":
    main()
