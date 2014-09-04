#!/usr/bin/env python
#from __future__ import print_function
import sys
import numpy as np
import time
import os.path
import matplotlib.pyplot as plt
from scipy import optimize
import pylab as plb

#bonds between cg sites
cg_bond_pairs = [["C1", "C2"], ["C2", "C3"], ["C3", "C4"], ["C4", "C5"],\
                 ["C5", "O5"], ["O5", "C1"]]

#bond angles between cg sites
cg_bond_triples = [["O5", "C1", "C2"], ["C1", "C2", "C3"], ["C2", "C3", "C4"],\
                   ["C3", "C4", "C5"], ["C4", "C5", "O5"], ["C5", "O5", "C1"]]

#bond dihedrals between cg sites
cg_bond_quads = [["O5", "C1", "C2", "C3"], ["C1", "C2", "C3", "C4"],\
                 ["C2", "C3", "C4", "C5"], ["C3", "C4", "C5", "O5"],\
                 ["C4", "C5", "O5", "C1"], ["C5", "O5", "C1", "C2"]]


def graph_output(output_all, request):
    rearrange = zip(*output_all)
    plt.figure()
    for i, item in enumerate(rearrange):
        plt.subplot(2,3, i+1)
        data = plt.hist(item, bins=100, normed=1)
        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma**2))
        x = [0.5*(data[1][j] + data[1][j+1]) for j in xrange(len(data[1])-1)]
        y = data[0]
        try:
            p0 = [np.max(y), x[np.argmax(y)], 0.1]
            popt, pcov = optimize.curve_fit(gauss, x, y, p0=p0)
            x_fit = plb.linspace(x[0], x[-1], 100)
            y_fit = gauss(x_fit, *popt)
            plt.plot(x_fit, y_fit, lw=4, color="r")
            print(popt)
        except RuntimeError:
            print("Failed to optimise fit")

if __name__ == "__main__":
    #frames = read(sys.argv[1])
    f = open("bond_lengths.csv", "r")
    cg_all_dists = []
    for line in f:
        try:
            tmp = [float(t) for t in line.split(",")]
            cg_all_dists.append(tmp)
        except:
            pass
    f.close()
    f = open("bond_angles.csv", "r")
    cg_all_angles = []
    for line in f:
        try:
            tmp = [float(t) for t in line.split(",")]
            cg_all_angles.append(tmp)
        except:
            pass
    f.close()
    cg_all_dihedrals = []
    f = open("bond_dihedrals.csv", "r")
    for line in f:
        try:
            tmp = [float(t) for t in line.split(",")]
            cg_all_dihedrals.append(tmp)
        except:
            pass
    f.close()
    np.set_printoptions(precision=3, suppress=True)
    graph_output(cg_all_dists, cg_bond_pairs)
    graph_output(cg_all_angles, cg_bond_triples)
    graph_output(cg_all_dihedrals, cg_bond_quads)
    plt.show()