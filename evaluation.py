import itertools
import json
from typing import Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from create_track import create_track
from trackers import tracker_soft, tracker_strong
from utils import NpEncoder


def create_report(df: pd.DataFrame, categories: set[str], metrics_list: list[str], report_name_suffix: str = ""):
    print(df.to_markdown())

    md_file_path = f"metrics_report_{report_name_suffix}.md" if report_name_suffix else "metrics_report.md"
    with open(md_file_path, "w") as f:
        f.write(df.to_markdown() + "\n\n")

    for category in categories:
        fig, ax = plt.subplots(1, len(metrics_list), figsize=(3 * len(metrics_list), 3))
        for i, metric_name in enumerate(metrics_list):
            df.boxplot(by=category, column=[metric_name], ax=ax[i])
        fig.tight_layout()
        figure_path = f"metrics_figures/boxplot_{category}_{report_name_suffix}.png"
        fig.savefig(figure_path)
        plt.show()

        with open(md_file_path, "a") as md_file:
            md_file.write(f"![Boxplot]({figure_path})\n")


class Monitor:
    def __init__(self):
        self.available_metrics = {
            "Mismatch ratio": self.mismatch_ratio,
            "Track fragmentation": self.track_fragmentation,
            "Track purity": self.track_purity,
            "Average track length": self.average_track_length,
        }
        self.tracks = {}

    def _preprocess_tracks(self):
        for key, track in self.tracks.items():
            self.tracks[key] = np.array(track).astype(int)

    def calculate_track_metrics(self) -> dict[str, float]:
        self._preprocess_tracks()
        return {metric_name: fun() for metric_name, fun in self.available_metrics.items()}

    def track_fragmentation(self) -> float:
        fragmentation_counts = [self._get_fragmentations(track) for track in self.tracks.values()]
        return np.mean(fragmentation_counts)

    def track_purity(self) -> float:
        purities = [self._get_purity(track) for track in self.tracks.values()]
        return np.mean(purities)

    def average_track_length(self) -> float:
        lengths = [len(track) for track in self.tracks.values()]
        return np.mean(lengths) if lengths else 0

    def mismatch_ratio(self) -> float:
        errors = np.sum([self._get_mismatch_errors(track) for track in self.tracks.values()])
        detections = np.sum([len(track) for track in self.tracks.values()])
        return 1 - (errors / detections)

    @staticmethod
    def _get_mismatch_errors(track: list[int]) -> int:
        if len(track) < 2:
            return 0
        track_array = np.array(track)
        mismatches = np.count_nonzero(track_array[1:] != track_array[:-1])
        return mismatches

    @staticmethod
    def _get_fragmentations(track: list[int]) -> int:
        return np.count_nonzero(np.diff(track) != 0)

    @staticmethod
    def _get_purity(track: list[int]) -> float:
        most_common_id_count = np.max(np.bincount(track))
        return most_common_id_count / len(track)

    def update(self, frame) -> None:
        for detection in filter(lambda d: d["bounding_box"], frame["data"]):
            if detection["cb_id"] not in self.tracks:
                self.tracks[detection["cb_id"]] = []
            self.tracks[detection["cb_id"]].append(detection["track_id"])


def soft_tracker_loop(track_data: Iterable, metrics_monitor: Monitor) -> Monitor:
    for el in track_data:
        if el["frame_id"] == 1:
            id_info = {}
            num = 0

        try:
            el_soft, id_info, num = tracker_soft(el, id_info, num)
            el_soft = json.loads(json.dumps(el_soft, cls=NpEncoder))
            metrics_monitor.update(el_soft)
        except IndexError:
            continue
    return metrics_monitor


def strong_tracker_loop(track_data: Iterable, metrics_monitor: Monitor) -> Monitor:
    for el in track_data:
        try:
            el_strong = tracker_strong(el)
            el_strong = json.loads(json.dumps(el_strong, cls=NpEncoder))
            metrics_monitor.update(el_strong)
        except Exception:
            continue
    return metrics_monitor


if __name__ == "__main__":
    tracks_amount_list = [5, 10, 20]
    random_range_list = [0, 10]
    skip_percent_list = [0.0, 0.05, 0.25, 0.5]

    tracks_amount_list = [5, 10]
    random_range_list = [0, 10]
    skip_percent_list = [0.0, 0.05]

    categorical_columns = {"tracks_amount", "random_range", "bb_skip_percent"}

    tracker_type = "soft"
    tracker_type = "strong"

    data = []
    for ta, rr, sp in tqdm(itertools.product(tracks_amount_list, random_range_list, skip_percent_list)):
        _, track_data = create_track(tracks_amount=ta, random_range=rr, bb_skip_percent=sp)

        metrics_monitor = Monitor()
        match tracker_type:
            case "soft":
                metrics_monitor = soft_tracker_loop(track_data, metrics_monitor)
            case "strong":
                metrics_monitor = strong_tracker_loop(track_data, metrics_monitor)

        series = pd.Series(
            {
                "tracks_amount": ta,
                "random_range": rr,
                "bb_skip_percent": sp,
                **metrics_monitor.calculate_track_metrics(),
            }
        )
        data.append(series)

    df = pd.DataFrame(data).round(3)
    metrics_list = list(metrics_monitor.available_metrics.keys())
    create_report(df, categorical_columns, metrics_list, tracker_type)
