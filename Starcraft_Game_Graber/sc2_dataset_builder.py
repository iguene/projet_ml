"""
SC2 Dataset Builder — Projet Annuel IA/BD
==========================================
Pipeline : .SC2Replay → snapshots temporels → features éco + heatmaps → CSV + NPY

Dépendances :
    pip install sc2reader numpy pandas matplotlib scipy tqdm

Usage :
    python sc2_dataset_builder.py --replays ./replays --output ./dataset
    python sc2_dataset_builder.py --replays ./replays --output ./dataset --timestamps 300 600 900
    python sc2_dataset_builder.py --replays ./replays --output ./dataset --multi-snapshot --interval 60
"""

import argparse
import json
import logging
import math
import os
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import sc2reader
from sc2reader.events import UnitBornEvent, UnitDiedEvent, UnitTypeChangeEvent
from tqdm import tqdm

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SC2Builder")

# ── Constantes ─────────────────────────────────────────────────────────────────
HEATMAP_SIZE = 32          # grille 32×32 (normalisée sur la carte)
MAP_BOUND    = 200.0       # coordonnée max typique SC2 (sera auto-détectée)

# Unités de combat pertinentes (exclus : workers, bâtiments, overlords)
WORKER_TYPES = {
    "SCV", "Probe", "Drone",
    "MULE", "Larva", "Egg",
}
SUPPLY_TYPES = {
    "Overlord", "Overseer", "OverlordTransport",
    "SupplyDepot", "SupplyDepotLowered", "Pylon",
}

# ── Structures de données ──────────────────────────────────────────────────────
@dataclass
class PlayerSnapshot:
    """Features économiques d'un joueur à un instant t."""
    player_id:        int   = 0
    race:             str   = "Unknown"
    minerals:         int   = 0
    vespene:          int   = 0
    minerals_spent:   int   = 0
    vespene_spent:    int   = 0
    supply_used:      int   = 0
    supply_cap:       int   = 0
    workers_count:    int   = 0
    army_count:       int   = 0
    unit_types_count: int   = 0   # diversité des types d'unités
    buildings_count:  int   = 0
    units_lost:       int   = 0   # unités mortes depuis le début
    # Ratios (calculés après)
    supply_ratio:     float = 0.0
    eco_ratio:        float = 0.0  # workers / supply_used
    army_ratio:       float = 0.0  # army / supply_used


@dataclass
class Snapshot:
    """Un snapshot complet (les deux joueurs) à un timestamp donné."""
    replay_file:  str  = ""
    timestamp_s:  int  = 0
    map_name:     str  = ""
    duration_s:   int  = 0
    label:        int  = 0    # 1 = P1 gagne, 2 = P2 gagne, 0 = draw
    p1: PlayerSnapshot = field(default_factory=PlayerSnapshot)
    p2: PlayerSnapshot = field(default_factory=PlayerSnapshot)
    # Deltas P1 - P2
    delta_minerals:  int   = 0
    delta_vespene:   int   = 0
    delta_workers:   int   = 0
    delta_army:      int   = 0
    delta_supply:    float = 0.0


# ── Helpers ────────────────────────────────────────────────────────────────────
def frames_to_seconds(frames: int, fps: float = 22.4) -> float:
    return frames / fps


def seconds_to_frames(seconds: float, fps: float = 22.4) -> int:
    return int(seconds * fps)


def is_worker(unit_type_name: str) -> bool:
    return any(w in unit_type_name for w in WORKER_TYPES)


def is_building(unit_type_name: str) -> bool:
    """Heuristique : nom sans espace, souvent suffixé par 'Structure' ou connu."""
    building_hints = [
        "Nexus", "Gateway", "CyberneticsCore", "Forge", "Stargate",
        "RoboticsFacility", "TemplarArchives", "DarkShrine",
        "CommandCenter", "OrbitalCommand", "PlanetaryFortress",
        "Barracks", "Factory", "Starport", "EngineeringBay",
        "Hatchery", "Lair", "Hive", "SpawningPool", "RoachWarren",
        "HydraliskDen", "UltraliskCavern", "Spire", "GreaterSpire",
        "Structure",
    ]
    return any(b in unit_type_name for b in building_hints)


# ── Heatmap ────────────────────────────────────────────────────────────────────
def build_heatmap(
    positions: list[tuple[float, float]],
    grid_size: int = HEATMAP_SIZE,
    map_size: tuple[float, float] = (MAP_BOUND, MAP_BOUND),
) -> np.ndarray:
    """
    Construit une heatmap (grid_size × grid_size) à partir d'une liste de (x, y).
    Chaque cellule contient le nombre d'unités présentes.
    """
    hmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    if not positions:
        return hmap

    w, h = map_size
    for x, y in positions:
        col = min(int((x / w) * grid_size), grid_size - 1)
        row = min(int((y / h) * grid_size), grid_size - 1)
        hmap[row, col] += 1.0

    # Normalisation par le max (évite les divergences entre parties de durées diff.)
    mx = hmap.max()
    if mx > 0:
        hmap /= mx
    return hmap


def smooth_heatmap(hmap: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Lissage gaussien optionnel pour des heatmaps plus douces."""
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(hmap, sigma=sigma)
    except ImportError:
        return hmap


# ── Extraction d'un replay ─────────────────────────────────────────────────────
def extract_replay(
    replay_path: Path,
    timestamps_s: list[int],
    grid_size:    int = HEATMAP_SIZE,
) -> tuple[list[Snapshot], dict[int, dict[int, np.ndarray]]]:
    """
    Parse un replay et retourne :
      - snapshots  : liste de Snapshot (un par timestamp)
      - heatmaps   : { timestamp_s → { player_id → heatmap } }

    Returns ([], {}) en cas d'erreur (replay corrompu, <2 joueurs, etc.)
    """
    try:
        replay = sc2reader.load_replay(str(replay_path), load_level=4)
    except Exception as e:
        log.warning(f"  Échec chargement {replay_path.name}: {e}")
        return [], {}

    # Vérifications de base
    if len(replay.players) < 2:
        log.warning(f"  {replay_path.name}: moins de 2 joueurs, ignoré.")
        return [], {}

    # Résultat de la partie
    p1_player = replay.players[0]
    p2_player = replay.players[1]
    duration_s = int(frames_to_seconds(replay.frames))

    # Label : 1 = P1 gagne, 2 = P2 gagne, 0 = draw
    if p1_player.result == "Win":
        label = 1
    elif p2_player.result == "Win":
        label = 2
    else:
        label = 0  # draw / unknown

    map_name = replay.map_name or "Unknown"

    # Taille de la carte (pour normaliser les coords)
    map_w = getattr(replay, "map_info", None)
    if map_w and hasattr(map_w, "width"):
        map_size = (float(replay.map_info.width), float(replay.map_info.height))
    else:
        map_size = (MAP_BOUND, MAP_BOUND)

    # ── Reconstruction état par frame ──────────────────────────────────────────
    # On reconstruit l'état des unités vivantes et les stats économiques
    # en rejouant les events jusqu'au timestamp cible.

    fps = 22.4

    # Suivi des unités vivantes : { unit_id → (owner_id, type_name, x, y) }
    alive_units: dict[int, tuple[int, str, float, float]] = {}
    dead_counts: dict[int, int] = {1: 0, 2: 0}

    # Stats économiques cumulées (sc2reader expose via tracker events)
    # On utilise les PlayerStatsEvent si disponibles
    eco_by_frame: dict[int, dict[int, dict]] = {}  # frame → pid → stats

    # Récolte les PlayerStatsEvents
    for event in replay.events:
        if event.__class__.__name__ == "PlayerStatsEvent":
            f = event.frame
            pid = event.player.pid
            if f not in eco_by_frame:
                eco_by_frame[f] = {}
            eco_by_frame[f][pid] = {
                "minerals":       getattr(event, "minerals_current", 0),
                "vespene":        getattr(event, "vespene_current", 0),
                "minerals_spent": getattr(event, "minerals_used_active_forces", 0)
                                + getattr(event, "minerals_used_economy", 0)
                                + getattr(event, "minerals_used_technology", 0),
                "vespene_spent":  getattr(event, "vespene_used_active_forces", 0)
                                + getattr(event, "vespene_used_economy", 0)
                                + getattr(event, "vespene_used_technology", 0),
                "supply_used":    getattr(event, "food_used", 0),
                "supply_cap":     getattr(event, "food_made", 0),
            }

    def get_eco_at(frame: int, pid: int) -> dict:
        """Retourne le dernier éco connu avant ou à `frame`."""
        best = {}
        for f in sorted(eco_by_frame.keys()):
            if f <= frame:
                if pid in eco_by_frame[f]:
                    best = eco_by_frame[f][pid]
            else:
                break
        return best

    # Positions des unités au frame cible : reconstruit event par event
    # On trie les timestamps pour ne parcourir les events qu'une fois
    sorted_ts = sorted(timestamps_s)
    valid_ts = [t for t in sorted_ts if t <= duration_s]

    snapshots: list[Snapshot] = []
    heatmaps: dict[int, dict[int, np.ndarray]] = {}

    # Index courant dans les events
    events_sorted = sorted(replay.events, key=lambda e: e.frame)
    event_idx = 0
    n_events = len(events_sorted)

    # État courant des unités
    # unit_id → { owner, type_name, x, y }
    units_state: dict[int, dict] = {}

    for ts in valid_ts:
        target_frame = seconds_to_frames(ts, fps)

        # Avance jusqu'au frame cible
        while event_idx < n_events:
            ev = events_sorted[event_idx]
            if ev.frame > target_frame:
                break
            event_idx += 1

            ename = ev.__class__.__name__

            if ename == "UnitBornEvent":
                uid = ev.unit.id
                try:
                    owner = ev.unit.owner.pid if ev.unit.owner else 0
                    x = float(ev.location[0]) if ev.location else 0.0
                    y = float(ev.location[1]) if ev.location else 0.0
                    units_state[uid] = {
                        "owner": owner,
                        "type":  ev.unit_type_name,
                        "x": x, "y": y,
                    }
                except Exception:
                    pass

            elif ename == "UnitDiedEvent":
                uid = ev.unit.id
                if uid in units_state:
                    owner = units_state[uid]["owner"]
                    if owner in dead_counts:
                        dead_counts[owner] += 1
                    del units_state[uid]

            elif ename in ("UnitPositionsEvent",):
                # Mise à jour des positions
                if hasattr(ev, "units"):
                    for unit, pos in ev.units:
                        uid = unit.id
                        if uid in units_state:
                            units_state[uid]["x"] = float(pos[0])
                            units_state[uid]["y"] = float(pos[1])

        # ── Calcul des features au timestamp ts ───────────────────────────────
        p_snaps: dict[int, PlayerSnapshot] = {}

        for pid_idx, player in enumerate(replay.players[:2]):
            pid = player.pid
            eco = get_eco_at(target_frame, pid)
            race = getattr(player, "play_race", "Unknown") or "Unknown"

            # Compte les unités par catégorie
            workers   = 0
            army      = 0
            buildings = 0
            unit_types: set[str] = set()
            positions_army:    list[tuple[float, float]] = []
            positions_all:     list[tuple[float, float]] = []

            for u in units_state.values():
                if u["owner"] != pid:
                    continue
                t = u["type"]
                pos = (u["x"], u["y"])
                positions_all.append(pos)
                unit_types.add(t)

                if is_building(t):
                    buildings += 1
                elif is_worker(t):
                    workers += 1
                else:
                    army += 1
                    positions_army.append(pos)

            sup_used = eco.get("supply_used", workers + army)
            sup_cap  = eco.get("supply_cap", 0)

            ps = PlayerSnapshot(
                player_id      = pid,
                race           = race,
                minerals       = eco.get("minerals", 0),
                vespene        = eco.get("vespene", 0),
                minerals_spent = eco.get("minerals_spent", 0),
                vespene_spent  = eco.get("vespene_spent", 0),
                supply_used    = sup_used,
                supply_cap     = sup_cap,
                workers_count  = workers,
                army_count     = army,
                unit_types_count = len(unit_types),
                buildings_count= buildings,
                units_lost     = dead_counts.get(pid, 0),
                supply_ratio   = (sup_used / sup_cap) if sup_cap > 0 else 0.0,
                eco_ratio      = (workers / sup_used) if sup_used > 0 else 0.0,
                army_ratio     = (army / sup_used) if sup_used > 0 else 0.0,
            )
            p_snaps[pid] = ps

            # Heatmaps
            if ts not in heatmaps:
                heatmaps[ts] = {}
            heatmaps[ts][pid] = smooth_heatmap(
                build_heatmap(positions_army, grid_size, map_size)
            )

        if len(p_snaps) < 2:
            continue

        ps1 = p_snaps[replay.players[0].pid]
        ps2 = p_snaps[replay.players[1].pid]

        snap = Snapshot(
            replay_file  = replay_path.name,
            timestamp_s  = ts,
            map_name     = map_name,
            duration_s   = duration_s,
            label        = label,
            p1           = ps1,
            p2           = ps2,
            delta_minerals = ps1.minerals - ps2.minerals,
            delta_vespene  = ps1.vespene  - ps2.vespene,
            delta_workers  = ps1.workers_count - ps2.workers_count,
            delta_army     = ps1.army_count    - ps2.army_count,
            delta_supply   = ps1.supply_ratio  - ps2.supply_ratio,
        )
        snapshots.append(snap)

    return snapshots, heatmaps


# ── Sérialisation ──────────────────────────────────────────────────────────────
def snapshot_to_row(snap: Snapshot) -> dict:
    """Aplatit un Snapshot en dict pour le CSV."""
    row = {
        "replay_file":  snap.replay_file,
        "timestamp_s":  snap.timestamp_s,
        "map_name":     snap.map_name,
        "duration_s":   snap.duration_s,
        "label":        snap.label,
    }
    for prefix, ps in [("p1", snap.p1), ("p2", snap.p2)]:
        for k, v in asdict(ps).items():
            row[f"{prefix}_{k}"] = v
    row["delta_minerals"] = snap.delta_minerals
    row["delta_vespene"]  = snap.delta_vespene
    row["delta_workers"]  = snap.delta_workers
    row["delta_army"]     = snap.delta_army
    row["delta_supply"]   = snap.delta_supply
    return row


# ── Pipeline principal ─────────────────────────────────────────────────────────
def build_dataset(
    replays_dir:  Path,
    output_dir:   Path,
    timestamps_s: list[int],
    grid_size:    int = HEATMAP_SIZE,
    max_replays:  Optional[int] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_files = sorted(replays_dir.glob("**/*.SC2Replay"))

    if not replay_files:
        log.error(f"Aucun fichier .SC2Replay trouvé dans {replays_dir}")
        return

    if max_replays:
        replay_files = replay_files[:max_replays]

    log.info(f"Replays trouvés : {len(replay_files)}")
    log.info(f"Timestamps cibles : {timestamps_s} s")
    log.info(f"Taille heatmap : {grid_size}×{grid_size}")

    all_rows: list[dict] = []
    # heatmaps_store[ts][pid] → liste de np.ndarray (une par replay)
    heatmaps_store: dict[int, dict[int, list[np.ndarray]]] = {
        ts: {1: [], 2: []} for ts in timestamps_s
    }
    errors = 0

    for rfile in tqdm(replay_files, desc="Parsing replays"):
        try:
            snaps, hmaps = extract_replay(rfile, timestamps_s, grid_size)

            for snap in snaps:
                all_rows.append(snapshot_to_row(snap))

            for ts, pid_maps in hmaps.items():
                for pid, hmap in pid_maps.items():
                    # pid dans le replay peut être 1 ou 2
                    key = 1 if pid == min(pid_maps.keys()) else 2
                    if ts in heatmaps_store and key in heatmaps_store[ts]:
                        heatmaps_store[ts][key].append(hmap)

        except Exception:
            errors += 1
            log.debug(traceback.format_exc())

    if not all_rows:
        log.error("Aucun snapshot extrait. Vérifiez vos replays.")
        return

    # ── Export CSV ─────────────────────────────────────────────────────────────
    csv_path = output_dir / "dataset.csv"
    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False)
    log.info(f"CSV exporté : {csv_path}  ({len(df)} lignes)")

    # ── Export heatmaps (.npy) ─────────────────────────────────────────────────
    # Format : (N_replays, grid_size, grid_size) par (timestamp, joueur)
    meta = {}
    for ts in timestamps_s:
        for pid_key in [1, 2]:
            hmlist = heatmaps_store[ts][pid_key]
            if hmlist:
                arr = np.stack(hmlist, axis=0)  # (N, H, W)
                npy_name = f"heatmap_t{ts}s_p{pid_key}.npy"
                npy_path = output_dir / npy_name
                np.save(npy_path, arr)
                meta[npy_name] = {"timestamp_s": ts, "player": pid_key, "shape": list(arr.shape)}
                log.info(f"Heatmap exportée : {npy_path}  shape={arr.shape}")

    # ── Métadonnées ────────────────────────────────────────────────────────────
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "timestamps_s":    timestamps_s,
            "grid_size":       grid_size,
            "total_snapshots": len(all_rows),
            "total_replays":   len(replay_files),
            "errors":          errors,
            "label_mapping":   {"0": "draw", "1": "player1_wins", "2": "player2_wins"},
            "heatmaps":        meta,
        }, f, indent=2)
    log.info(f"Métadonnées : {meta_path}")

    # ── Résumé ─────────────────────────────────────────────────────────────────
    log.info("─" * 50)
    log.info(f"✓ Replays traités : {len(replay_files) - errors}/{len(replay_files)}")
    log.info(f"✓ Snapshots totaux : {len(all_rows)}")
    log.info(f"✓ Distribution labels : {df['label'].value_counts().to_dict()}")
    log.info(f"✓ Output : {output_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="SC2 Dataset Builder — PA IA/BD")
    parser.add_argument("--replays",  type=Path, required=True,
                        help="Dossier contenant les .SC2Replay")
    parser.add_argument("--output",   type=Path, default=Path("./dataset"),
                        help="Dossier de sortie (défaut: ./dataset)")
    parser.add_argument("--timestamps", type=int, nargs="+",
                        default=[300, 600, 900],
                        help="Timestamps fixes en secondes (défaut: 300 600 900)")
    parser.add_argument("--multi-snapshot", action="store_true",
                        help="Ajoute des snapshots réguliers sur toute la durée")
    parser.add_argument("--interval", type=int, default=60,
                        help="Intervalle en secondes pour --multi-snapshot (défaut: 60)")
    parser.add_argument("--max-duration", type=int, default=2400,
                        help="Durée max couverte par --multi-snapshot (défaut: 2400s)")
    parser.add_argument("--grid-size", type=int, default=HEATMAP_SIZE,
                        help=f"Taille de la grille heatmap (défaut: {HEATMAP_SIZE})")
    parser.add_argument("--max-replays", type=int, default=None,
                        help="Limite le nombre de replays traités (debug)")
    return parser.parse_args()


def main():
    args = parse_args()

    timestamps = list(args.timestamps)

    if args.multi_snapshot:
        extra = list(range(args.interval, args.max_duration + 1, args.interval))
        timestamps = sorted(set(timestamps + extra))
        log.info(f"Mode multi-snapshot activé : {len(timestamps)} timestamps de {timestamps[0]}s à {timestamps[-1]}s")

    build_dataset(
        replays_dir  = args.replays,
        output_dir   = args.output,
        timestamps_s = timestamps,
        grid_size    = args.grid_size,
        max_replays  = args.max_replays,
    )


if __name__ == "__main__":
    main()
