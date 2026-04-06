# SC2 Dataset Builder — PA IA/BD

Pipeline de collecte et traitement de replays StarCraft II pour la prédiction de résultat de match.

## Installation

```bash
pip install -r requirement.txt
```

## Structure de sortie

```
dataset/
├── dataset.csv          ← features tabulaires (un snapshot par ligne)
├── heatmap_t300s_p1.npy ← heatmaps joueur 1 à t=300s  (N, 32, 32)
├── heatmap_t300s_p2.npy ← heatmaps joueur 2 à t=300s  (N, 32, 32)
├── heatmap_t600s_p1.npy
├── heatmap_t600s_p2.npy
├── ...
└── metadata.json        ← config + statistiques du run
```

## Modes d'utilisation

### Timestamps fixes (recommandé pour commencer)
```bash
python sc2_dataset_builder.py \
    --replays ./replays \
    --output  ./dataset \
    --timestamps 300 600 900   # snapshots à 5min, 10min, 15min
```

### Multi-snapshot (série temporelle)
```bash
python sc2_dataset_builder.py \
    --replays ./replays \
    --output  ./dataset \
    --multi-snapshot \
    --interval 60 \
    --max-duration 1800
```

### Debug (limiter le nombre de replays)
```bash
python sc2_dataset_builder.py --replays ./replays --output ./dataset --max-replays 10
```

## Format du CSV

| Colonne | Description |
|---|---|
| `label` | 0=draw, 1=P1 gagne, 2=P2 gagne |
| `timestamp_s` | instant du snapshot (secondes) |
| `p1_minerals` | ressources minérales P1 |
| `p1_vespene` | ressources vespène P1 |
| `p1_army_count` | unités de combat P1 |
| `p1_workers_count` | workers P1 |
| `p1_supply_ratio` | supply_used / supply_cap |
| `p1_army_ratio` | army / supply_used |
| `p1_units_lost` | unités mortes P1 depuis début |
| `delta_minerals` | p1_minerals - p2_minerals |
| `delta_army` | p1_army - p2_army |
| `delta_supply` | p1_supply_ratio - p2_supply_ratio |
| ... | (symétrique pour p2) |

## Format des heatmaps (.npy)

```python
import numpy as np
arr = np.load("dataset/heatmap_t600s_p1.npy")
# shape : (N_replays, 32, 32)
# valeurs : [0.0, 1.0] — densité normalisée des unités de combat
```

## Inspection du dataset

```bash
python inspect_dataset.py --dataset ./dataset
python inspect_dataset.py --dataset ./dataset --heatmap heatmap_t600s_p1.npy --idx 5
```

Génère automatiquement :
- `eco_distributions.png` — distributions des features par label
- `feature_correlations.png` — corrélation de Pearson avec le label
- `heatmap_sample_*.png` — visualisation d'une heatmap

## Intégration ML (exemple)

```python
import numpy as np
import pandas as pd

df = pd.read_csv("dataset/dataset.csv")

# Features tabulaires
feature_cols = [c for c in df.columns if c not in
                ("label", "replay_file", "map_name", "p1_race", "p2_race")]
X_tabular = df[feature_cols].values
y = df["label"].values

# Heatmaps (à fusionner avec X_tabular ou entrée CNN séparée)
hmap_p1 = np.load("dataset/heatmap_t600s_p1.npy")  # (N, 32, 32)
hmap_p2 = np.load("dataset/heatmap_t600s_p2.npy")
hmap_diff = hmap_p1 - hmap_p2                       # carte différentielle

# Aplatir pour MLP/SVM
X_heatmap = hmap_diff.reshape(len(hmap_diff), -1)   # (N, 1024)
```

## Notes

- Les `.SC2Replay` sont lus récursivement dans le dossier `--replays`
- Les replays corrompus ou à moins de 2 joueurs sont ignorés silencieusement
- La heatmap encode uniquement les **unités de combat** (armée), pas les workers ni bâtiments parce que j'en suis pas à ce niveau pour l'instant
- Un lissage gaussien (σ=1) est appliqué aux heatmaps pour réduire le bruit et éviter ainsi les faux positif
- Beaucoup d'améliorations possible et à prévoir, pour l'instant ça consistitue un dataset donc on va s'arrêter à là avant de
revenir pour apoprter des features d'analyses et de traitement plus pousser au dataset
- Ce README a été généré par IA sous conseil de mon tuteur parce que c'était impossible de me faire comprendre autrement