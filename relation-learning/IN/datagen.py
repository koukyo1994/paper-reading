import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import constants as const

from pathlib import Path
from typing import Tuple, Union


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
        self.objects = np.zeros([n_steps, const.FEATURE_DIM, n_objects],
                                dtype=float)

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
        rr = np.zeros([self.n_steps, self.n_objects, n_relations],
                      dtype=np.float32)
        rs = np.zeros([self.n_steps, self.n_objects, n_relations],
                      dtype=np.float32)
        ra = np.zeros([self.n_steps, n_attributes, n_relations],
                      dtype=np.float32)
        return rr, rs, ra

    def _create_externals(self, n_externals: int) -> np.ndarray:
        x = np.zeros([self.n_steps, n_externals, self.n_objects], dtype=int)
        return x

    def to_tensor(self):
        objects = torch.from_numpy(self.objects)
        externals = torch.from_numpy(self.externals)
        triplets = tuple(torch.from_numpy(t) for t in self.triplets)
        return objects, externals, triplets

    def to_mp4(self, savedir: Union[Path, str], name: str):
        if isinstance(savedir, Path):
            filename = savedir / name
        else:
            filename = Path(savedir) / name

        colors = ["r", "b", "g", "k", "y", "m", "c"]
        out_img_fnames = []
        with tempfile.TemporaryDirectory() as td:
            dirname = Path(td)
            for i in range(self.n_steps):
                fig = plt.figure(figsize=(3, 3))
                ax = fig.add_subplot(111)
                ax.set_xlim(-200, 200)
                ax.set_ylim(-200, 200)
                for j in range(self.n_objects):
                    ax.scatter(
                        self.objects[i, 1, j],
                        self.objects[i, 0, j],
                        c=colors[j % len(colors)],
                        marker="o")
                ax.axis("off")
                fig.savefig(dirname / f"{i}.png")
                out_img_fnames.append(str(dirname / f"{i}.png"))
                plt.close(fig)

            tmp = cv2.imread(out_img_fnames[0])
            IMG_SIZE = tmp.shape[0]

            fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            video = cv2.VideoWriter(
                str(filename), fourcc, 20.0, (IMG_SIZE, IMG_SIZE))

            for img_file_names in out_img_fnames:
                img = cv2.imread(img_file_names)
                video.write(img)
            video.release()


class GravitySimulation(Simulation):
    def __init__(self,
                 n_objects: int,
                 timestep: float,
                 n_steps: int,
                 orbital=True):
        self.orbital = orbital

        super().__init__(n_objects, timestep, n_steps)

    def _prepare_n_relations(self):
        return self.n_objects * (self.n_objects - 1)

    def _prepare_n_externals(self):
        return 0

    def _prepare_n_attributes(self):
        return 1

    def _create_relation_triplets(self, n_relations: int, n_attributes: int):
        rr, rs, ra = super()._create_relation_triplets(n_relations,
                                                       n_attributes)
        cnt = 0
        for i in range(self.n_objects):
            for j in range(self.n_objects):
                if i != j:
                    rr[:, i, cnt] = 1
                    rs[:, j, cnt] = 1
                    cnt += 1
        return rr, rs, ra

    def _create_objects(self, objects: np.ndarray) -> np.ndarray:
        # initialize object position and velocity
        if self.orbital:
            # Create a static object in the middle with 100 unit mass
            # set mass
            self.objects[0, 0, 0] = 100
            # set position and velocities
            self.objects[0, 1:const.FEATURE_DIM, 0] = 0.0

            for i in range(1, self.n_objects):
                # set random mass
                self.objects[0, 0, i] = np.random.rand() * const._MASS_COEFS[0] + \
                    const._MASS_COEFS[1]

                distance = np.random.rand(
                ) * const._DISTANCE_COEFS[0] + const._DISTANCE_COEFS[1]
                theta = np.pi / 2 - np.radians(np.random.rand() * 360)

                # set random position
                self.objects[0, 1, i] = distance * np.cos(theta)
                self.objects[0, 2, i] = distance * np.sin(theta)

                # set velocity according to the first object
                self.objects[0, 3, i] = -1 * self.objects[0, 2, i] / self.norm(
                    self.objects[0, 1:3, i]) * (
                        const.G * self.objects[0, 0, 0] / self.norm(
                            self.objects[0, 1:3, i])**2) * distance / 1000
                self.objects[0, 4, i] = self.objects[0, 1, i] / self.norm(
                    self.objects[0, 1:3, i]) * (
                        const.G * self.objects[0, 0, 0] / self.norm(
                            self.objects[0, 1:3, i])**2) * distance / 1000
        else:
            for i in range(self.n_objects):
                # set random mass
                self.objects[0, 0, i] = np.random.rand() * const._MASS_COEFS[0] + \
                    const._MASS_COEFS[1]

                distance = np.random.rand(
                ) * const._DISTANCE_COEFS[0] + const._DISTANCE_COEFS[1]
                theta = np.pi / 2 - np.radians(np.random.rand() * 360)

                # set random position
                self.objects[0, 1, i] = distance * np.cos(theta)
                self.objects[0, 2, i] = distance * np.sin(theta)

                # set random velocity
                self.objects[0, 3, i] = np.random.rand() * const._VELOCITY_COEFS[0] - \
                    const._VELOCITY_COEFS[1]
                self.objects[0, 4, i] = np.random.rand() * const._VELOCITY_COEFS[0] - \
                    const._VELOCITY_COEFS[1]

        for step in range(1, self.n_steps):
            current_state = self.objects[step - 1]

            next_state = np.zeros((const.FEATURE_DIM, self.n_objects),
                                  dtype=float)
            force_mat = np.zeros((self.n_objects, self.n_objects, 2),
                                 dtype=float)
            force_sum = np.zeros((self.n_objects, 2), dtype=float)
            acceleration = np.zeros((self.n_objects, 2), dtype=float)

            for i in range(self.n_objects):
                for j in range(i + 1, self.n_objects):
                    if j != i:
                        force = self.force(current_state[:3, i],
                                           current_state[:3, j])
                        force_mat[i, j] += force
                        force_mat[j, i] -= force
                force_sum[i] = np.sum(force_mat[i], axis=0)
                # F = ma
                acceleration[i] = force_sum[i] / current_state[0, i]

                # Copy mass - always constant
                next_state[0, i] = current_state[0, i]
                next_state[3:5, i] = current_state[
                    3:5, i] + acceleration[i] * self.timestep
                next_state[1:3, i] = current_state[
                    1:3, i] + next_state[3:5, i] * self.timestep
            self.objects[step] = next_state
        return self.objects

    @staticmethod
    def norm(x: np.ndarray):
        return np.sqrt(np.sum(x**2))

    def force(self, reciever: np.ndarray, sender: np.ndarray):
        diff = sender[1:3] - reciever[1:3]
        distance = self.norm(diff)
        if distance < 1:
            distance = 1
        return const.G * reciever[0] * sender[0] / (distance**3) * diff


if __name__ == "__main__":
    sim = GravitySimulation(n_objects=3, timestep=1e-3, n_steps=1000)
    objects, externals, triplets = sim.to_tensor()

    print("objects size: ", objects.size())
    print("externals size: ", externals.size())
    print("triplet first elem size: ", triplets[0].size())
    print("triplet second elem size: ", triplets[1].size())
    print("triplet third elem size: ", triplets[2].size())

    sim.to_mp4(savedir="data", name="test.mp4")
