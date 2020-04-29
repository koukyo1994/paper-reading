import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path
from typing import Tuple, Union

from .constants import FEATURE_DIM


class Simulation:
    def __init__(self, n_objects: int, timestep: float, n_steps: int):
        self.n_steps = n_steps
        self.n_objects = n_objects
        self.timestep = timestep

        # dummy
        n_relations = self._prepare_n_relations()
        n_attributes = self._prepare_n_attributes()
        n_externals = self._prepare_n_externals()

        # initialization
        self.objects = np.zeros([n_steps, FEATURE_DIM, n_objects], dtype=float)

        # simulation
        self.objects = self._create_objects(self.objects)
        self.triplets = self._create_relation_triplets(n_relations,
                                                       n_attributes)
        self.externals = self._create_externals(n_externals)

    def _prepare_n_relations(self) -> int:
        return self.n_objects

    def _prepare_n_attributes(self) -> int:
        return self.n_objects

    def _prepare_n_externals(self) -> int:
        return self.n_objects

    def _create_objects(self, objects: np.ndarray):
        return objects

    def _create_relation_triplets(
            self, n_relations: int,
            n_attributes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rr = np.zeros(self.n_objects, n_relations, dtype=int)
        rs = np.zeros(self.n_objects, n_relations, dtype=int)
        ra = np.zeros(n_attributes, n_relations, dtype=float)
        return rr, rs, ra

    def _create_externals(self, n_externals: int) -> np.ndarray:
        x = np.zeros(n_externals, self.n_objects, dtype=int)
        return x

    def to_tensor(self):
        objects = torch.from_numpy(self.objects)
        externals = torch.from_numpy(self.externals)
        triplets = (torch.from_numpy(t) for t in self.triplets)
        return objects, externals, triplets

    def to_mp4(self, savedir: Union[Path, str], name: str):
        metadata = {
            "title": f"{name} animation",
            "artist": "Hidehisa Arai",
            "comment": ""
        }
        writer = animation.writers["ffmpeg"](fps=15, metadata=metadata)
        fig = plt.figure(figsize=(3, 3))
        plt.xlim(-200, 200)
        plt.ylim(-200, 200)
        n_figures = len()