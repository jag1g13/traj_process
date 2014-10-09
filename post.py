#!/usr/bin/env python
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


def graph_output_time(output_all, num=0):
    rearrange = zip(*output_all)
    plt.figure()
    if num == 0:
        for i, item in enumerate(rearrange):
            #locs, labels = plt.xticks()
            plt.xticks([])
            #plt.setp(labels, rotation=90)
            plt.subplot(2,3,i+1)
            data = plt.plot(item)
    else:
        data = plt.plot(rearrange[num])

def graph_dipole_time(dipoles_all, num=-1, part=2):
    rearrange = [[],[],[],[],[],[]]
    for frame in dipoles_all:
        for i, atom in enumerate(frame):
            rearrange[i].append(atom[part])
    plt.figure()
    if num == -1:
        for i, item in enumerate(rearrange):
            #locs, labels = plt.xticks()
            plt.xticks([])
            #plt.setp(labels, rotation=90)
            plt.subplot(2,3,i+1)
            data = plt.plot(item)
    else:
        data = plt.plot(rearrange[num])

def graph_dipole(dipoles_all, num=-1, part=2):
    rearrange = [[],[],[],[],[],[]]
    for frame in dipoles_all:
        for i, atom in enumerate(frame):
            rearrange[i].append(atom[part])
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
            locs, labels = plt.xticks()
            plt.yticks([])
            plt.setp(labels, rotation=90)
            p0 = [np.max(y), x[np.argmax(y)], 0.1]
            popt, pcov = optimize.curve_fit(gauss, x, y, p0=p0)
            x_fit = plb.linspace(x[0], x[-1], 100)
            y_fit = gauss(x_fit, *popt)
            plt.plot(x_fit, y_fit, lw=4, color="r")
            print("G: ", popt)
        except RuntimeError:
            print("Failed to optimise fit")


def graph_output(output_all, print_raw=1):
    rearrange = zip(*output_all)
    #fig = plt.figure()
    fig, ax = plt.subplots(2,3)
    fig.tight_layout()
    for i, item in enumerate(rearrange):
        ax1 = plt.subplot(2,3, i+1)
        data = ax1.hist(item, bins=100, normed=1)
        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma**2))
        def harmonic(x, *p):
            a, b, c = p
            return a * (x-b)*(x-b) + c
        k_B = 1.3806e-23
        T = 300
        x = [0.5*(data[1][j] + data[1][j+1]) for j in xrange(len(data[1])-1)]
        y = data[0]
        if not print_raw:
            plt.cla()
        try:
            locs, labels = plt.xticks()
            plt.setp(labels, rotation=90)
            #ax1.xticks(x, labels, rotation=90)
            #plt.yticks([])
            p0 = [np.max(y), x[np.argmax(y)], 0.1]
            popt, pcov = optimize.curve_fit(gauss, x, y, p0=p0)
            #popt[0] = np.abs(popt[0])
            A, mu, sigma = popt
            popt = A, mu, np.abs(sigma)
            sigma = np.abs(sigma)
            x_fit = plb.linspace(x[0], x[-1], 100)
            y_fit = gauss(x_fit, *popt)
            ax1.plot(x_fit, y_fit, lw=2, color="r")
            print("G: ", popt)
            #start doing boltzmann inversion
            if(A<0 or sigma<0):
                print("AAAAARGH!!!!!!")
            g_i = (A / (sigma*np.sqrt(np.pi/2))) * np.exp(-2 * (x_fit - mu)*(x_fit - mu) / (sigma*sigma))
            #print(y_fit)
            #print(g_i)
            #y_inv = -k_B * T * np.log(g_i)
            y_inv = np.log(g_i)
            #print(y_inv)
            p0 = [-1, x_fit[np.argmax(y_inv)], np.max(y_inv)]
            popt, pcov = optimize.curve_fit(harmonic, x_fit, y_inv, p0=p0)
            popt[2] = 0
            popt[0] = np.abs(popt[0])
            y_inv_fit = harmonic(x_fit, *popt)
            #plt.plot(x_fit, y_inv, lw=4, color="b")
            ax2 = ax1.twinx()
            ax2.plot(x_fit, y_inv_fit, lw=2, color="y")
            print("H: ", popt)
        except RuntimeError:
            print("Failed to optimise fit")


def boltzmann_inversion(x_fit, y_fit):
    """
    do a boltzmann inversion on the fitted gaussian to obtain a harmonic potential
    """
    k_B = 1.3806e-23
    T = 300
    y_inv = -k_B * T * np.log(y_fit / (x_fit*x_fit))
    plt.plot(x_fit, y_inv, lw=2, color="r")
    def harmonic(x, *p):
        a, b = p
        return a * (x-b)*(x-b)
    p0 = [1, x_fit[np.argmax(y_fit)]]
    popt, pcov = optimize.curve_fit(harmonic, x_fit, y_inv, p0=p0)
    y_inv_fit = harmonic(x_fit, *popt)
    plt.plot(x_fit, y_inv_fit, lw=2, color="b")
    #plb.savefig("dists_inv.pdf", bbox_inches="tight")
    plt.show()
    return
    

def auto(dists, angles, dihedrals, dipoles):
    print("Dists")
    graph_output(dists)
    plb.savefig("dists.pdf", bbox_inches="tight")
    print("Angles")
    graph_output(angles)
    plb.savefig("angles.pdf", bbox_inches="tight")
    print("Dihedrals")
    graph_output(dihedrals)
    plb.savefig("dihedrals.pdf", bbox_inches="tight")
    graph_output_time(dists)
    plb.savefig("dists_time.pdf", bbox_inches="tight")
    graph_output_time(angles)
    plb.savefig("angles_time.pdf", bbox_inches="tight")
    graph_output_time(dihedrals)
    plb.savefig("dihedrals_time.pdf", bbox_inches="tight")
    print("Dipoles_0")
    graph_dipole(dipoles, part=0)
    plb.savefig("dipoles_0.pdf", bbox_inches="tight")
    graph_dipole_time(dipoles, part=0)
    plb.savefig("dipoles_time_0.pdf", bbox_inches="tight")
    print("Dipoles_1")
    graph_dipole(dipoles, part=1)
    plb.savefig("dipoles_1.pdf", bbox_inches="tight")
    graph_dipole_time(dipoles, part=1)
    plb.savefig("dipoles_time_1.pdf", bbox_inches="tight")
    print("Dipoles_2")
    graph_dipole(dipoles, part=2)
    plb.savefig("dipoles_2.pdf", bbox_inches="tight")
    graph_dipole_time(dipoles, part=2)
    plb.savefig("dipoles_time_2.pdf", bbox_inches="tight")
    

if __name__ == "__main__":
    f = open("bond_lengths.csv", "r")
    dists = []
    for line in f:
        try:
            tmp = [float(t) for t in line.split(",")]
            dists.append(np.array(tmp))
        except:
            pass
    f.close()
    f = open("bond_angles.csv", "r")
    angles = []
    for line in f:
        try:
            tmp = [float(t) for t in line.split(",")]
            angles.append(np.array(tmp))
        except:
            pass
    f.close()
    dihedrals = []
    f = open("bond_dihedrals.csv", "r")
    for line in f:
        try:
            tmp = [float(t) for t in line.split(",")]
            dihedrals.append(np.array(tmp))
        except:
            pass
    f.close()
    dipoles = []
    f = open("dipoles.csv", "r")
    i = 0
    for line in f:
        try:
            tmp = [float(t) for t in line.split(",")]
            if i%6 == 0:
                dipoles.append([])
            dipoles[-1].append(np.array(tmp))
            i += 1
        except:
            pass
    f.close()
    np.set_printoptions(precision=3, suppress=True)
    print("Ready for commands")
    while True:
        s = raw_input(">>>")
        if s=="auto":
            auto(dists, angles, dihedrals, dipoles)
            break
        if s=="help":
            print(help_msg)
        else:
            eval(s)