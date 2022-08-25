import dataclasses
from collections import namedtuple
import numpy as np
from typing import Tuple


class ParticleCloud(object):
    """
    This class is a thin wrapper for particles of a given type
    """
    def __init__(self, n_particles: int, initial_coordinate: Tuple[float, float, float, float]):
        self.x = np.random.rand(n_particles) * (initial_coordinate[1] - initial_coordinate[0]) + initial_coordinate[0]
        self.y = np.random.rand(n_particles) * (initial_coordinate[3] - initial_coordinate[2]) + initial_coordinate[2]

    def average_x(self) -> float:
        return np.average(self.x)

    def average_y(self) -> float:
        return np.average(self.y)
