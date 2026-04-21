#!/usr/bin/env python3
"""Convert raw NGSIM trajectory text files into TrajPred HDF5 files.

The original repository does not include its preprocessing pipeline. This script
creates the minimum schema needed by the MMnTP config:

    state_merging, output_states_data, labels, frame_data, tv_data

Feature columns are documented in STATE_MERGING_COLUMNS below. They are a
transparent baseline, not a claim to reproduce the original authors' features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


FT_TO_M = 0.3048
STATE_MERGING_COLUMNS = [
    "lat_vel",
    "lon_vel",
    "lat_acc",
    "lon_acc",
    "local_x",
    "local_y",
    "lane_id",
    "vehicle_length",
    "vehicle_width",
    "vehicle_class",
    "preceding_present",
    "preceding_rel_x",
    "preceding_rel_y",
    "preceding_rel_vx",
    "preceding_rel_vy",
    "following_present",
    "following_rel_x",
    "following_rel_y",
    "following_rel_vx",
    "following_rel_vy",
    "left_front_rel_y",
    "left_back_rel_y",
    "right_front_rel_y",
    "right_back_rel_y",
    "space_headway",
    "time_headway",
    "lane_change_sign",
]

NGSIM_COLUMNS = [
    "vehicle_id",
    "frame_id",
    "total_frames",
    "global_time",
    "local_x",
    "local_y",
    "global_x",
    "global_y",
    "vehicle_length",
    "vehicle_width",
    "vehicle_class",
    "vehicle_velocity",
    "vehicle_acceleration",
    "lane_id",
    "preceding",
    "following",
    "space_headway",
    "time_headway",
]


def read_ngsim(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"[\s,]+",
        header=None,
        names=NGSIM_COLUMNS,
        engine="python",
    )
    df = df.dropna(how="all")
    if df.shape[1] != len(NGSIM_COLUMNS):
        raise ValueError(f"Expected {len(NGSIM_COLUMNS)} NGSIM columns in {path}")
    return df


def prepare_tracks(df: pd.DataFrame, frame_stride: int) -> pd.DataFrame:
    df = df.copy()
    min_frame = int(df["frame_id"].min())
    df = df[(df["frame_id"] - min_frame) % frame_stride == 0].copy()

    for col in [
        "local_x",
        "local_y",
        "global_x",
        "global_y",
        "vehicle_length",
        "vehicle_width",
        "vehicle_velocity",
        "vehicle_acceleration",
        "space_headway",
    ]:
        df[col] = df[col].astype(float) * FT_TO_M

    df = df.sort_values(["vehicle_id", "frame_id"]).reset_index(drop=True)
    frame_step = int(frame_stride)
    new_segment = (
        (df["vehicle_id"] != df["vehicle_id"].shift(1))
        | ((df["frame_id"] - df["frame_id"].shift(1)) != frame_step)
    )
    df["segment_id"] = new_segment.cumsum().astype(np.int64)
    df["tv_id"] = df["vehicle_id"].astype(np.int64) * 1000 + df["segment_id"]

    dt = frame_stride / 10.0
    df["prev_local_x"] = df.groupby("segment_id")["local_x"].shift(1)
    df["prev_local_y"] = df.groupby("segment_id")["local_y"].shift(1)
    df["prev_lane_id"] = df.groupby("segment_id")["lane_id"].shift(1)
    df["lat_vel"] = ((df["local_x"] - df["prev_local_x"]) / dt).fillna(0.0)
    df["lon_vel"] = df["vehicle_velocity"].fillna(0.0)
    df["lat_acc"] = df.groupby("segment_id")["lat_vel"].diff().fillna(0.0) / dt
    df["lon_acc"] = df["vehicle_acceleration"].fillna(0.0)
    lane_delta = (df["lane_id"] - df["prev_lane_id"]).fillna(0).astype(int)
    df["label"] = np.select([lane_delta > 0, lane_delta < 0], [1, 2], default=0)
    df["lane_change_sign"] = np.sign(lane_delta).astype(float)
    df["dx"] = (df["local_x"] - df["prev_local_x"]).fillna(0.0)
    df["dy"] = (df["local_y"] - df["prev_local_y"]).fillna(0.0)
    return df


def row_lookup(df: pd.DataFrame) -> dict[tuple[int, int], pd.Series]:
    return {
        (int(row.frame_id), int(row.vehicle_id)): row
        for row in df.itertuples(index=False)
    }


def add_direct_neighbor_features(
    row: pd.Series, lookup: dict[tuple[int, int], pd.Series], neighbor_col: str
) -> list[float]:
    neighbor_id = int(row[neighbor_col])
    neighbor = lookup.get((int(row["frame_id"]), neighbor_id))
    if neighbor_id == 0 or neighbor is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        1.0,
        float(neighbor.local_x - row["local_x"]),
        float(neighbor.local_y - row["local_y"]),
        float(neighbor.lat_vel - row["lat_vel"]),
        float(neighbor.lon_vel - row["lon_vel"]),
    ]


def adjacent_lane_features(frame_df: pd.DataFrame, row: pd.Series) -> list[float]:
    lane = int(row["lane_id"])
    y = float(row["local_y"])
    values = []
    for adj_lane in (lane - 1, lane + 1):
        candidates = frame_df[frame_df["lane_id"] == adj_lane]
        ahead = candidates[candidates["local_y"] > y]
        behind = candidates[candidates["local_y"] < y]
        front_rel_y = (
            float((ahead["local_y"] - y).min()) if not ahead.empty else 0.0
        )
        back_rel_y = (
            float((behind["local_y"] - y).max()) if not behind.empty else 0.0
        )
        values.extend([front_rel_y, back_rel_y])
    return values


def build_arrays(df: pd.DataFrame, min_track_len: int) -> dict[str, np.ndarray]:
    lookup = row_lookup(df)
    frames = {frame: frame_df for frame, frame_df in df.groupby("frame_id")}
    state_rows: list[list[float]] = []
    output_rows: list[list[float]] = []
    label_rows: list[int] = []
    frame_rows: list[int] = []
    tv_rows: list[int] = []

    for _, group in df.groupby("segment_id", sort=False):
        if len(group) < min_track_len:
            continue
        for _, row in group.iterrows():
            preceding = add_direct_neighbor_features(row, lookup, "preceding")
            following = add_direct_neighbor_features(row, lookup, "following")
            adjacent = adjacent_lane_features(frames[int(row["frame_id"])], row)
            features = [
                float(row["lat_vel"]),
                float(row["lon_vel"]),
                float(row["lat_acc"]),
                float(row["lon_acc"]),
                float(row["local_x"]),
                float(row["local_y"]),
                float(row["lane_id"]),
                float(row["vehicle_length"]),
                float(row["vehicle_width"]),
                float(row["vehicle_class"]),
                *preceding,
                *following,
                *adjacent,
                float(row["space_headway"]),
                float(row["time_headway"]),
                float(row["lane_change_sign"]),
            ]
            if len(features) != len(STATE_MERGING_COLUMNS):
                raise AssertionError("state_merging feature count mismatch")
            state_rows.append(features)
            output_rows.append([float(row["dx"]), float(row["dy"])])
            label_rows.append(int(row["label"]))
            frame_rows.append(int(row["frame_id"]))
            tv_rows.append(int(row["tv_id"]))

    if not state_rows:
        raise ValueError("No valid NGSIM tracks remained after filtering")

    return {
        "state_merging": np.asarray(state_rows, dtype=np.float32),
        "output_states_data": np.asarray(output_rows, dtype=np.float32),
        "labels": np.asarray(label_rows, dtype=np.int64),
        "frame_data": np.asarray(frame_rows, dtype=np.int64),
        "tv_data": np.asarray(tv_rows, dtype=np.int64),
    }


def write_h5(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        for key, value in arrays.items():
            h5.create_dataset(key, data=value)
        h5.attrs["state_merging_columns"] = ",".join(STATE_MERGING_COLUMNS)
        h5.attrs["source"] = "scripts/convert_ngsim_to_h5.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/archive/trajectories-0750am-0805am.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed_ngsim/RenderedDataset"),
    )
    parser.add_argument("--file-id", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--min-in-seq-len", type=int, default=15)
    parser.add_argument("--tgt-seq-len", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    min_track_len = args.min_in_seq_len + args.tgt_seq_len
    df = prepare_tracks(read_ngsim(args.input), args.frame_stride)
    arrays = build_arrays(df, min_track_len=min_track_len)
    output_path = args.output_dir / f"{args.file_id:02d}.h5"
    write_h5(output_path, arrays)
    print(f"Wrote {output_path}")
    for key, value in arrays.items():
        print(f"{key}: {value.shape} {value.dtype}")


if __name__ == "__main__":
    main()
