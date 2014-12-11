import numpy as np
import matplotlib.pyplot as plt
# from scipy import optimize
from math import sqrt

# from frame import Frame


class FieldMap:
    """
    Calclates, stores and manipulates the electric field around a molecule, given from a Frame instance
    """
    def __init__(self):
        self.border = 1       # disance from molecule that the grid extends in nm
        # self.grid_dim = [3, 75, 75]       # looks okay as an image
        # self.grid_dim = [3, 125, 125]     # looks good as an image
        self.grid_dim = [5, 5, 5]     # looks bad as an image
        self.grid = np.zeros(self.grid_dim)
        print("Grid:", self.grid_dim)
        print(str(self.grid_dim[0] * self.grid_dim[1] * self.grid_dim[2]) + " grid points")

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
        self.grid_x = np.linspace(xmin, xmax, self.grid_dim[0])
        self.grid_y = np.linspace(ymin, ymax, self.grid_dim[1])
        self.grid_z = np.linspace(zmin, zmax, self.grid_dim[2])

    def calc_field(self, frame):
        inveps = 1. / (4 * np.pi * 8.854187817e-12)
        # inveps = 8.9875517873681e9
        for i in xrange(self.grid_dim[0]):
            for j in xrange(self.grid_dim[1]):
                for k in xrange(self.grid_dim[2]):
                    self.grid[i][j][k] = 0.
                    for atom in frame.atoms:
                        self.grid[i][j][k] += atom.charge / self.dist_sqr(atom, i, j, k)
                    # self.grid[i][j][k] *= inveps

    def dist_sqr(self, atom, i, j, k):
        return (atom.loc[0]-self.grid_x[i])**2 + \
               (atom.loc[1]-self.grid_y[j])**2 + \
               (atom.loc[2]-self.grid_z[k])**2

    def plot(self, x):
        # plt.contourf(self.grid_y, self.grid_z, self.grid[x])
        plt.pcolor(self.grid_y, self.grid_z, self.grid[x], vmin=-1, vmax=1)
        # plt.imshow(self.grid[x], extent=[self.grid_y[0], self.grid_y[-1], self.grid_z[0], self.grid_z[-1]])
        plt.show()

    def __repr__(self):
        # print("Grid:", self.grid_dim)
        # print(self.grid)
        return "Grid: " + repr(self.grid_dim) + "\n" +\
               str(self.grid_dim[0] * self.grid_dim[1] * self.grid_dim[2]) + " grid points\n" +\
               repr(self.grid)
