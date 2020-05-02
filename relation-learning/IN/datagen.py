import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

import constants as const

from pathlib import Path
from typing import Tuple, Union

from scipy.io import loadmat, savemat
from scipy.sparse import csr_matrix


class Simulation:
    def __init__(self,
                 n_objects: int = 3,
                 timestep: float = 0.001,
                 n_steps: int = 1000):
        self.n_steps = n_steps
        self.n_objects = n_objects
        self.timestep = timestep

        # dummy
        self.n_relations = self._prepare_n_relations()
        self.n_attributes = self._prepare_n_attributes()
        self.n_externals = self._prepare_n_externals()

        # initialization
        self.objects = np.zeros([n_steps, const.FEATURE_DIM, n_objects],
                                dtype=np.float32)
        self.triplets = (np.zeros([n_steps, self.n_objects, self.n_relations],
                                  dtype=np.float32),
                         np.zeros([n_steps, self.n_objects, self.n_relations],
                                  dtype=np.float32),
                         np.zeros(
                             [n_steps, self.n_attributes, self.n_relations],
                             dtype=np.float32))
        self.externals = np.zeros([n_steps, self.n_externals, self.n_objects],
                                  dtype=np.float32)

    def simulate(self):
        self.objects = self._create_objects(self.objects)
        self.triplets = self._create_relation_triplets(self.n_relations,
                                                       self.n_attributes)
        self.externals = self._create_externals(self.n_externals)

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
        x = np.zeros([self.n_steps, n_externals, self.n_objects],
                     dtype=np.float32)
        return x

    def to_tensor(self):
        objects = torch.from_numpy(self.objects)
        externals = torch.from_numpy(self.externals)
        triplets = tuple(torch.from_numpy(t) for t in self.triplets)
        return objects, externals, triplets

    def to_mp4(self, savedir: Union[Path, str], name: str):
        if isinstance(savedir, Path):
            savedir.mkdir(exist_ok=True, parents=True)
            filename = savedir / name
        else:
            Path(savedir).mkdir(exist_ok=True, parents=True)
            filename = Path(savedir) / name

        colors = ["r", "b", "g", "k", "y", "m", "c"]
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111)
        ax.set(xlim=(-200, 200), ylim=(-200, 200))
        ax.axis("off")
        # initialize
        N_TRAJECTORY = 20
        points = []
        for j in range(self.n_objects):
            for _ in range(N_TRAJECTORY):
                p, = ax.plot([], [], "o", color=colors[j % len(colors)])
                points.append(p)

        def init():
            return tuple(points)

        def update(step: int):
            for j in range(self.n_objects):
                offset = j * N_TRAJECTORY
                for o in range(N_TRAJECTORY):

                    alpha = 1 / N_TRAJECTORY * (o + 1)

                    index = step - (N_TRAJECTORY - o - 1)
                    if index < 0:
                        index = 0
                    data = self.objects[index, :, j]
                    x, y = data[1], data[2]

                    points[offset].set_data(x, y)
                    points[offset].set_alpha(alpha)

                    offset += 1
            return points

        ani = animation.FuncAnimation(
            fig, update, frames=self.n_steps, interval=10, init_func=init)
        ani.save(filename)

    def save(self, savedir: Union[Path, str], name: str):
        if isinstance(savedir, Path):
            savedir.mkdir(exist_ok=True, parents=True)
            filename = savedir / name
        else:
            Path(savedir).mkdir(exist_ok=True, parents=True)
            filename = Path(savedir) / name
        objects = csr_matrix(self.objects.flatten())
        externals = csr_matrix(self.externals.flatten())
        triplets = tuple(csr_matrix(t.flatten()) for t in self.triplets)

        state_dict = {
            "objects": objects,
            "externals": externals,
            "triplets": triplets,
            "n_steps": self.n_steps,
            "n_objects": self.n_objects,
            "n_relations": self.n_relations,
            "timestep": self.timestep
        }
        savemat(str(filename), state_dict)

    def load(self, path: Union[Path, str]):
        state_dict = loadmat(str(path))

        n_steps = state_dict["n_steps"][0, 0]
        n_objects = state_dict["n_objects"][0, 0]
        n_relations = state_dict["n_relations"][0, 0]
        timestep = state_dict["timestep"][0, 0]

        self.objects = state_dict["objects"].toarray().reshape(
            n_steps, -1, n_objects)
        self.externals = state_dict["externals"].toarray().reshape(
            n_steps, -1, n_objects)
        self.triplets = tuple(  # type: ignore
            t.toarray().reshape(n_steps, -1, n_relations)
            for t in state_dict["triplets"][0])

        self.n_steps = n_steps
        self.n_objects = n_objects
        self.n_relations = n_relations
        self.timestep = timestep
        self.n_attributes = self.triplets[2].shape[1]


class GravitySimulation(Simulation):
    def __init__(self,
                 n_objects: int = 3,
                 timestep: float = 0.001,
                 n_steps: int = 1000,
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
                                  dtype=np.float32)
            force_mat = np.zeros((self.n_objects, self.n_objects, 2),
                                 dtype=np.float32)
            force_sum = np.zeros((self.n_objects, 2), dtype=np.float32)
            acceleration = np.zeros((self.n_objects, 2), dtype=np.float32)

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

    sim.simulate()

    objects, externals, triplets = sim.to_tensor()

    print("objects size: ", objects.size())
    print("externals size: ", externals.size())
    print("triplet first elem size: ", triplets[0].size())
    print("triplet second elem size: ", triplets[1].size())
    print("triplet third elem size: ", triplets[2].size())

    sim.to_mp4(savedir="data", name="test.mp4")
    sim.save(savedir="data", name="test.mat")

    del sim

    sim = GravitySimulation(n_objects=5, timestep=1e-3, n_steps=1000)
    sim.load("data/test.mat")
    print(sim.objects.shape)
