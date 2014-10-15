#!/usr/bin/env python
import sys
import numpy as np
import time
import os.path
import matplotlib.pyplot as plt
from scipy import optimize
import pylab as plb
from optparse import OptionParser
import cProfile
import pstats
import multiprocessing

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


def graph_output_time(output_all, filename, num=0):
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
    plb.savefig(filename+"_time.pdf", bbox_inches="tight")
    plt.close()

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
    plb.savefig("dipoles_time_"+str(part)+".pdf", bbox_inches="tight")
    plt.close()

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
    plb.savefig("dipoles_"+str(part)+".pdf", bbox_inches="tight")
    plt.close()


def graph_output(output_all, filename, print_raw=1):
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
            popt[2] = np.abs(popt[2])
            A, mu, sigma = popt
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
    plb.savefig(filename+".pdf", bbox_inches="tight")
    plt.close()


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
    pool = multiprocessing.Pool(4)
    for i in [[dists, "dists"], [angles, "angles"], [dihedrals, "dihedrals"]]:
        pool.apply_async(graph_output(i[0], i[1]))
        #graph_output(i[0], i[1]))
        pool.apply_async(graph_output_time(i[0], i[1]))
        #graph_output_time(i[0], i[1]))
    for i in [0,1,2]:
        pool.apply_async(graph_dipole(dipoles, i))
        #graph_dipole(dipoles, i))
        pool.apply_async(graph_dipole_time(dipoles, i))
        #graph_dipole_time(dipoles, i))
    pool.close()
    pool.join()


def process_all(do_auto):
    t_start = time.clock()
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
    if do_auto:
        auto(dists, angles, dihedrals, dipoles)
    else:
        print("Ready for commands")
        while True:
            command = raw_input(">>>")
            eval(command)
    t_end = time.clock()
    print("\rCalculated {0} frames in {1}s\n".format(len(dists), (t_end - t_start)) + "-"*20)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--auto",
                      action="store_true", dest="auto", default=False,
                      help="Plot everything automatically")
    (options, args) = parser.parse_args()
    #process_all(options.auto)
    cProfile.run("process_all(options.auto)", "profile")
    p = pstats.Stats("profile")
    p.sort_stats('cumulative').print_stats(15)
    #p.sort_stats('time').print_stats(15)