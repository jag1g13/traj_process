import numpy as np
# from scipy import optimize
from math import sqrt

# from frame import Frame


class FieldMap:
    """
    Calclates, stores and manipulates the electric field around a molecule, given from a Frame instance
    """
    def __init__(self):
        self.border = 0.5       # disance from molecule that the grid extends in nm

    def setup_grid(self, frame):
        xmax, ymax, zmax = float("-inf"), float("-inf"), float("-inf")
        xmin, ymin, zmin = float("inf"), float("inf"), float("inf")
        for atom in frame.atoms:
            xmax = max(xmax, atom.loc[0])
            xmin = min(xmin, atom.loc[0])
            ymax = max(ymax, atom.loc[1])
            ymin = min(ymin, atom.loc[1])
            zmax = max(zmax, atom.loc[2])
            zmin = min(zmin, atom.loc[2])
        xmax += self.border
        ymax += self.border
        zmax += self.border
        xmin -= self.border
        ymin -= self.border
        zmin -= self.border
        print(xmin, xmax)
        print(ymin, ymax)
        print(zmin, zmax)
        size = 3
        # grid = np.mgrid[xmin:xmax:size, ymin:ymax:size, zmin:zmax:size]
        grid = np.zeros([size, size, size])
        print(grid)

    def calc_field(self, x, y, z):
        pass
