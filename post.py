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
import pyradi.ryplot as ryplot
from matplotlib import cm



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
    rearrange = zip(*output_all[::100])
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


def graph_dipole_time_3d(dipoles_all, num=-1):
    r = [[], [], [], [], [], []]
    theta = [[],[],[],[],[],[]]
    phi = [[],[],[],[],[],[]]
    for frame in dipoles_all[::100]:
        for i, atom in enumerate(frame):
            r[i].append(atom[0])
            theta[i].append(atom[1])
            phi[i].append(atom[2])
    #create the wireframe for the sphere
    u = np.linspace(0, np.pi, 18)
    v = np.linspace(0, 2 * np.pi, 18)
    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))
    for i in xrange(len(theta)):
        #avg_r = np.mean(r[i])
        avg_r = 1
        #create normal vectors using the pairs of angles in a transformation 
        xs = r[i] * np.cos(phi[i]) * np.cos(theta[i])
        ys = r[i] * np.cos(phi[i]) * np.sin(theta[i])
        zs = r[i] * np.sin(phi[i])
        azim = 45 # view angle
        elev = 45 # view angle
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.azim = azim
        ax.elev = elev
        ax.plot_wireframe(avg_r*x, avg_r*y, avg_r*z, color="y")
        ax.scatter(xs, ys, zs)
        max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
        mean_x = xs.mean()
        mean_y = ys.mean()
        mean_z = zs.mean()
        ax.set_xlim(mean_x - max_range, mean_x + max_range)
        ax.set_ylim(mean_y - max_range, mean_y + max_range)
        ax.set_zlim(mean_z - max_range, mean_z + max_range)
        plb.savefig("dipoles_time_3d_"+str(i)+".pdf", bbox_inches="tight")
    plt.show()
    plt.close()


def graph_dipole_time(dipoles_all, num=-1, part=2):
    rearrange = [[],[],[],[],[],[]]
    mag = [[], [], [], [], [], []]
    #col = range(len(dipoles_all)/100)
    for frame in dipoles_all[::100]:
        for i, atom in enumerate(frame):
            rearrange[i].append(atom[part])
            mag[i].append(atom[0])
    plt.figure()
    if num == -1:
        for i, item in enumerate(rearrange):
            #locs, labels = plt.xticks()
            #plt.setp(labels, rotation=90)
            plt.subplot(2,3,i+1, polar=True)
            #data = plt.plot(item)
            data = plt.scatter(item, mag[i])
    else:
        data = plt.plot(rearrange[num])
    plb.savefig("dipoles_time_"+str(part)+".pdf", bbox_inches="tight")
    plt.close()

def graph_dipole(dipoles_all, num=-1, part=2):
    global export
    if export:
        f = open("dipoles_"+str(part)+"_fit.csv", "a")
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
            if export:
                np.savetxt(f, popt, delimiter=",")
        except RuntimeError:
            print("Failed to optimise fit")
    plb.savefig("dipoles_"+str(part)+".pdf", bbox_inches="tight")
    plt.close()
    if export:
        f.close()


def graph_output(output_all, filename, print_raw=1):
    global export
    if export:
        f = open(filename+"_fit.csv", "a")
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
            if export:
                np.savetxt(f, popt, delimiter=",")
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
            if export:
                np.savetxt(f, popt, delimiter=",")
        except RuntimeError:
            print("Failed to optimise fit")
    plb.savefig(filename+".pdf", bbox_inches="tight")
    plt.close()
    if export:
        f.close()


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
    

def auto(dists, angles, dihedrals, dipoles, only_dipoles):
    #pool = multiprocessing.Pool(4)
    if not only_dipoles:
        for i in [[dists, "dists"], [angles, "angles"], [dihedrals, "dihedrals"]]:
            #pool.apply_async(graph_output(i[0], i[1]))
            graph_output(i[0], i[1])
            #pool.apply_async(graph_output_time(i[0], i[1]))
            graph_output_time(i[0], i[1])
        for i in [0,1,2]:
            #pool.apply_async(graph_dipole(dipoles, part=i))
            graph_dipole(dipoles, part=i)
            #pool.apply_async(graph_dipole_time(dipoles, part=i))
            graph_dipole_time(dipoles, part=i)
    graph_dipole_time_3d(dipoles)
    #pool.close()
    #pool.join()


def process_all(do_auto, only_dipoles):
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
        auto(dists, angles, dihedrals, dipoles, only_dipoles)
    else:
        print("Ready for commands")
        while True:
            command = raw_input(">>>")
            eval(command)
    t_end = time.clock()
    print("\rCalculated {0} frames in {1}s\n".format(len(dists), (t_end - t_start)) + "-"*20)


if __name__ == "__main__":
    global export
    parser = OptionParser()
    parser.add_option("-a", "--auto",
                      action="store_true", dest="auto", default=False,
                      help="Plot everything automatically")
    parser.add_option("-d", "--dipoles",
                      action="store_true", dest="only_dipoles", default=False,
                      help="Automatically do only dipoles")
    parser.add_option("-e", "--export",
                      action="store_true", dest="export", default=False,
                      help="Save fitting parameters")
    (options, args) = parser.parse_args()
    export = options.export
    process_all(options.auto, options.only_dipoles)
    #cProfile.run("process_all(options.auto, options.only_dipoles)", "profile")
    #p = pstats.Stats("profile")
    #p.sort_stats('cumulative').print_stats(15)
    #p.sort_stats('time').print_stats(15)