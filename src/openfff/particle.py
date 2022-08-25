import dataclasses
from collections import namedtuple
import numpy as np
from typing import Tuple
from constants import boltzmann_constant


class ParticleCloud(object):
    """
    This class is a thin wrapper for particles of a given type
    """

    def __init__(self, n_particles: int,
                 initial_coordinate: Tuple[float, float, float, float],
                 particle_diameter: float):
        """
        Create a particle cloud containing n_particles all located within the bounding box.
        :param n_particles: number of particles in the cloud
        :param initial_coordinate: the bounding box where the particles will be located initially
        :param particle_diameter: the diameter of these particles
        """
        # the x and y coordinates of each particle in the particle cloud
        self.x = np.random.rand(n_particles) * (initial_coordinate[1] - initial_coordinate[0]) + initial_coordinate[0]
        self.y = np.random.rand(n_particles) * (initial_coordinate[3] - initial_coordinate[2]) + initial_coordinate[2]

        # characteristics of particle cloud
        self.particle_diameter = particle_diameter

    def average_x(self) -> float:
        return np.average(self.x)

    def average_y(self) -> float:
        return np.average(self.y)

    def get_diffusion_coef(self, temp: float, dynamic_visc: float):
        """
        Calculates the diffusion coefficient for the given type of particle in this cloud
        :param temp: Temperature (K) at which to calculate the diffusion coefficient at
        :param dynamic_visc: The dynamic viscosity (Pa*s) of the medium the PC is located in
        :return: the diffusion coefficient in m/s
        """
        return (temp * boltzmann_constant) / (3 * np.pi * dynamic_visc * self.particle_diameter)
