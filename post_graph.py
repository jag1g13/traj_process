import sys
import numpy as np
import time
import os.path
import matplotlib.pyplot as plt
from scipy import optimize
import pylab as plb
from matplotlib import cm


# bonds between cg sites
cg_bond_pairs = [["C1", "C2"], ["C2", "C3"], ["C3", "C4"], ["C4", "C5"],
                 ["C5", "O5"], ["O5", "C1"]]

# bond angles between cg sites
cg_bond_triples = [["O5", "C1", "C2"], ["C1", "C2", "C3"], ["C2", "C3", "C4"],
                   ["C3", "C4", "C5"], ["C4", "C5", "O5"], ["C5", "O5", "C1"]]

# bond dihedrals between cg sites
cg_bond_quads = [["O5", "C1", "C2", "C3"], ["C1", "C2", "C3", "C4"],
                 ["C2", "C3", "C4", "C5"], ["C3", "C4", "C5", "O5"],
                 ["C4", "C5", "O5", "C1"], ["C5", "O5", "C1", "C2"]]

adjacent = {"C1": ["O5", "C2"], "C2": ["C1", "C3"], "C3": ["C2", "C4"],
            "C4": ["C3", "C5"], "C5": ["C4", "O5"], "O5": ["C5", "C1"]}

num_to_plot = 1000


def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


def harmonic(x, *p):
    a, b, c = p
    return a * (x-b)*(x-b) + c

#   def harmonic(x, *p):
#       a, b = p
#       return a * (x-b)*(x-b)


def graph_output_time(output_all, filename, num=0):
    rearrange = zip(*output_all[::int(len(output_all)/num_to_plot)+1])
    plt.figure()
    if num == 0:
        for i, item in enumerate(rearrange):
            plt.xticks([])
            plt.subplot(2, 3, i+1)
            data = plt.plot(item)
    else:
        data = plt.plot(rearrange[num])
    plb.savefig(filename+"_time.pdf", bbox_inches="tight")
    plt.close()


def graph_dipole_3d(dipoles_all, dists, angles, only_3d, num=-1):
    global num_to_plot
    r = [[], [], [], [], [], []]
    theta = [[], [], [], [], [], []]
    phi = [[], [], [], [], [], []]
    r_atoms = [[], [], [], [], [], []]
    theta_atoms = [[], [], [], [], [], []]
    for frame in dipoles_all[::int(len(dipoles_all)/num_to_plot)+1]:
        for i, atom in enumerate(frame):
            r[i].append(atom[0])
            theta[i].append(atom[1])
            phi[i].append(atom[2])
    rearrange_dists = zip(*dists)
    rearrange_angles = zip(*angles)
    for i in range(len(dipoles_all[0])):
        r_atoms[i] = np.average(r[i])
#       r_atoms[i].append(np.average(rearrange_dists[(i-1)%6]))
#       r_atoms[i].append(np.average(rearrange_dists[(i+1)%6]))
#       r_atoms[i] = [1.,1.]
        theta_tmp = np.pi * np.average(rearrange_angles[i]) / 180.
        theta_atoms[i] = [-theta_tmp, theta_tmp]
#   create the wireframe for the sphere
    u = np.linspace(0, np.pi, 18)
    v = np.linspace(0, 2 * np.pi, 18)
    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))
    for i in xrange(len(theta)):
        avg_r = np.mean(r[i])
#       avg_r = 1
#       create normal vectors using the pairs of angles in a transformation
        xs = r[i] * np.cos(phi[i]) * np.cos(theta[i])
        ys = r[i] * np.cos(phi[i]) * np.sin(theta[i])
        zs = r[i] * np.sin(phi[i])
        azim = 45   # view angle
        elev = 45   # view angle
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.azim = azim
        ax.elev = elev
        ax.plot_wireframe(avg_r*x, avg_r*y, avg_r*z, color="y")
        ax.scatter(xs, ys, zs)
#       now plot the locations of atoms
        xs_atom = r_atoms[i] * np.cos(theta_atoms[i])
        ys_atom = r_atoms[i] * np.sin(theta_atoms[i])
        zs_atom = np.zeros_like(ys_atom)
        ax.scatter(xs_atom, ys_atom, zs_atom, color="r", s=200)
        max_range = np.array([xs.max()-xs.min(),
                              ys.max()-ys.min(),
                              zs.max()-zs.min()]).max() / 2.0
        mean_x = xs.mean()
        mean_y = ys.mean()
        mean_z = zs.mean()
        ax.set_xlim(mean_x - max_range, mean_x + max_range)
        ax.set_ylim(mean_y - max_range, mean_y + max_range)
        ax.set_zlim(mean_z - max_range, mean_z + max_range)
        plb.savefig("dipoles_3d_"+str(i)+".pdf", bbox_inches="tight")
    if only_3d:
        plt.show()
    plt.close()


def graph_dipole_polar(dipoles_all, num=-1, part=2):
    global num_to_plot
    rearrange = [[], [], [], [], [], []]
    mag = [[], [], [], [], [], []]
#   for frame in dipoles_all:
    for frame in dipoles_all[::int(len(dipoles_all)/num_to_plot)+1]:
        for i, atom in enumerate(frame):
            rearrange[i].append(atom[part])
            mag[i].append(atom[0])
    plt.figure()
    if num == -1:
        for i, item in enumerate(rearrange):
            plt.subplot(2, 3, i+1, polar=True)
#           locs, labels = plt.xticks()
#           plt.setp(labels, rotation=90)
#           data = plt.plot(item)
            data = plt.scatter(item, mag[i])
    else:
        data = plt.plot(rearrange[num])
    plb.savefig("dipoles_polar_"+str(part)+".pdf", bbox_inches="tight")
    plt.close()


def graph_dipole_time(dipoles_all, num=-1, part=2):
    rearrange = [[], [], [], [], [], []]
    mag = [[], [], [], [], [], []]
#   for frame in dipoles_all:
    for frame in dipoles_all[::int(len(dipoles_all)/num_to_plot)+1]:
        for i, atom in enumerate(frame):
            rearrange[i].append(atom[part])
            mag[i].append(atom[0])
    plt.figure()
    if num == -1:
        for i, item in enumerate(rearrange):
            plt.subplot(2, 3, i+1)
#           locs, labels = plt.xticks()
#           plt.setp(labels, rotation=90)
            data = plt.plot(item)
#           data = plt.scatter(item, mag[i])
    else:
        data = plt.plot(rearrange[num])
    plb.savefig("dipoles_time_"+str(part)+".pdf", bbox_inches="tight")
    plt.close()


def graph_dipole(dipoles_all, num=-1, part=2, export=True):
    if export:
        f = open("dipoles_"+str(part)+"_fit.csv", "a")
    rearrange = [[], [], [], [], [], []]
    for frame in dipoles_all:
        for i, atom in enumerate(frame):
            rearrange[i].append(atom[part])
    plt.figure()
    for i, item in enumerate(rearrange):
        plt.subplot(2, 3, i+1)
        data = plt.hist(item, bins=100, normed=1)
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


def graph_output(output_all, filename, print_raw=1, export=True):
    if export:
        f = open(filename+"_fit.csv", "a")
    rearrange = zip(*output_all)
#   fig = plt.figure()
    fig, ax = plt.subplots(2, 3)
    fig.tight_layout()
#   np.seterr(all='raise')
    np.seterr(under='ignore')
    for i, item in enumerate(rearrange):
        ax1 = plt.subplot(2, 3, i+1)
        data = ax1.hist(item, bins=100, normed=1)
        k_B = 1.3806e-23
        R = 8.314
        T = 300
        x = [0.5*(data[1][j] + data[1][j+1]) for j in xrange(len(data[1])-1)]
        y = data[0]
        if not print_raw:
            plt.cla()
        try:
            print("-"*10)
            locs, labels = plt.xticks()
            plt.setp(labels, rotation=90)
            p0 = [np.max(y), x[np.argmax(y)], 0.1]
            popt, pcov = optimize.curve_fit(gauss, x, y, p0=p0)
            popt[2] = np.abs(popt[2])
            A, mu, sigma = popt
#           print(A, mu, sigma)
            x_fit = plb.linspace(x[0], x[-1], 100)
            y_fit = gauss(x_fit, *popt)
            ax1.plot(x_fit, y_fit, lw=2, color="r")
            print("G: ", popt)
            if export:
                np.savetxt(f, popt, delimiter=",")
#           start doing boltzmann inversion
            if(A <= 0 or sigma <= 0):
                print("AAAAARGH!!!!!!")
                raise FloatingPointError
            g_i = (A / (sigma*np.sqrt(np.pi/2))) *\
                np.exp(-2 * (x_fit - mu)*(x_fit - mu) / (sigma*sigma))
#           y_inv = -k_B * T * np.log(g_i)
            y_inv = R * T * np.log(g_i)
            p0 = [-1, x_fit[np.argmax(y_inv)], np.max(y_inv)]
            popt, pcov = optimize.curve_fit(harmonic, x_fit, y_inv, p0=p0)
            popt[2] = 0
            popt[0] = np.abs(popt[0])
            y_inv_fit = harmonic(x_fit, *popt)
            ax2 = ax1.twinx()
            ax2.plot(x_fit, y_inv_fit, lw=2, color="y")
            print("H: ", popt)
            if export:
                np.savetxt(f, popt, delimiter=",")
        except RuntimeError:
            print("Failed to optimise fit or perform inversion")
    plb.savefig(filename+".pdf", bbox_inches="tight")
    plt.close()
    if export:
        f.close()


def boltzmann_inversion(x_fit, y_fit):
    """
    do a boltzmann inversion on the fitted gaussian
    get a harmonic potential out
    """
    k_B = 1.3806e-23
    R = 8.314
    T = 300
#   y_inv = -k_B * T * np.log(y_fit / (x_fit*x_fit))
#   this gives per molecule values and is so small it doesn't print
    y_inv = -R * T * np.log(y_fit / (x_fit*x_fit))
    plt.plot(x_fit, y_inv, lw=2, color="r")
    p0 = [1, x_fit[np.argmax(y_fit)]]
    popt, pcov = optimize.curve_fit(harmonic, x_fit, y_inv, p0=p0)
    y_inv_fit = harmonic(x_fit, *popt)
    plt.plot(x_fit, y_inv_fit, lw=2, color="b")
#   plb.savefig("dists_inv.pdf", bbox_inches="tight")
    plt.show()
    return


def graph_energy(energies):
    plt.plot(energies)
    plb.savefig("energies.pdf", bbox_inches="tight")
    plt.close()


def auto(dists, angles, dihedrals, dipoles, energies, do_dipoles):
    if do_dipoles:
        for i in [0, 1, 2]:
            print("dipoles_"+str(i))
            graph_dipole(dipoles, part=i)
            graph_dipole_polar(dipoles, part=i)
            graph_dipole_time(dipoles, part=i)
        graph_dipole_3d(dipoles, dists, angles, only_3d)
    else:
        # graph_energy(energies)
        for i in [[dists, "dists"],
                  [angles, "angles"],
                  [dihedrals, "dihedrals"]]:
            print(i[1])
            graph_output(i[0], i[1])
            graph_output_time(i[0], i[1])


def process_all(do_auto, do_dipoles=False):
    t_start = time.clock()
    f = open("length.csv", "r")
    dists = []
    for line in f:
        try:
            tmp = [float(t) for t in line.split(",")]
            dists.append(np.array(tmp))
        except:
            pass
    f.close()
    f = open("angle.csv", "r")
    angles = []
    for line in f:
        try:
            tmp = [float(t) for t in line.split(",")]
            angles.append(np.array(tmp))
        except:
            pass
    f.close()
    dihedrals = []
    f = open("dihedral.csv", "r")
    for line in f:
        try:
            tmp = [float(t) for t in line.split(",")]
            dihedrals.append(np.array(tmp))
        except:
            pass
    f.close()
    dipoles = []
    if do_dipoles:
        f = open("dipoles.csv", "r")
        i = 0
        for line in f:
            try:
                tmp = [float(t) for t in line.split(",")]
                if i % 6 == 0:
                    dipoles.append([])
                dipoles[-1].append(np.array(tmp))
                i += 1
            except:
                pass
        f.close()
    # f = open("energies.csv", "r")
    energies = []
    # for line in f:
    #     energies.append(float(line))
    # f.close()
    np.set_printoptions(precision=3, suppress=True)
    if do_auto:
        auto(dists, angles, dihedrals, dipoles, energies, do_dipoles)
    else:
        print("Ready for commands")
        while True:
            command = raw_input(">>>")
            eval(command)
    t_end = time.clock()
    print("\rCalculated {0} frames in {1}s\n"
          .format(len(dists), (t_end - t_start)) + "-"*20)
