import numpy as np


class ElectricField:
    def __init__(self, magnitude: float):
        """
        :param magnitude: The magnitude of the electric field in V/m
        """
        self.E = magnitude

    def get_component_y(self, em: float) -> float:
        """
        :param em: The electrophoretic mobility of a specific particle in m^2/(V*s)
        :return: The velocity component in the y direction
        """
        return em * self.E
