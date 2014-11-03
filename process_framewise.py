#!/usr/bin/env python

"""
TODO
1/ Improve solvent RDF
2/ WTFDipoles?
3/ Remove this huge data section from the top
    -read structure from .gro file, calculate everything else
    -probably just need a mapping dictionary
"""

import sys
import numpy as np
import time
import os.path
import matplotlib.pyplot as plt
import MDAnalysis.coordinates.xdrfile.libxdrfile2 as xdr
import MDAnalysis.core.AtomGroup as AtomGroup
from scipy import optimize
import pylab as plb
import cProfile
import pstats
from math import sqrt
from optparse import OptionParser

verbose = False
cm_map = False

sugar_atom_nums = {}

sugar_atoms = ["C1", "O1", "HO1", "C2", "O2", "HO2",
               "C3", "O3", "HO3", "C4", "O4", "HO4",
               "C5", "O5", "C6",  "O6", "HO6"]

# atomic charges
atomic_charges = {"C4": 0.232,  "O4": -0.642, "HO4": 0.410, "C3": 0.232,
                  "O3": -0.642, "HO3": 0.410, "C2": 0.232,  "O2": -0.642,
                  "HO2": 0.410, "C6": 0.232,  "O6": -0.642, "HO6": 0.410,
                  "C5": 0.376,  "O5": -0.480, "C1": 0.232,  "O1": -0.538,
                  "HO1": 0.410, "HW1": 0.41,  "HW2": 0.41,  "OW": -0.82}

# bond dihedrals we want to know
bond_quads = []

# names of the cg sites
cg_sites = ["C1", "C2", "C3", "C4", "C5", "O5"]

# atom numbers of the cg sites
cg_atom_nums = {}

# which atoms map to which cg site, same order as above
cg_internal_map = {"C1": ["C1", "O1", "HO1"], "C2": ["C2", "O2", "HO2"],
                   "C3": ["C3", "O3", "HO3"], "C4": ["C4", "O4", "HO4"],
                   "C5": ["C5", "C6", "O6", "HO6"], "O5": ["O5"]}

cg_map = [["C1"], ["C2"], ["C3"],
          ["C4"], ["C5"], ["O5"]]

# remade this as a dictionary in preparation for the new coarse graining method
cg_map_new = {"C1": ["C1"], "C2": ["C2"], "C3": ["C3"],
              "C4": ["C4"], "C5": ["C5"], "O5": ["O5"],
              "OW": ["OW"]}

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

# adjacent sites, really just a remapped version of cg_bond_pairs
adjacent = {"C1": ["O5", "C2"], "C2": ["C1", "C3"], "C3": ["C2", "C4"],
            "C4": ["C3", "C5"], "C5": ["C4", "O5"], "O5": ["C5", "C1"]}

# bonds within a cg site
cg_internal_bonds = {"C1": [["C1", "O1"], ["O1", "HO1"]],
                     "C2": [["C2", "O2"], ["O2", "HO2"]],
                     "C3": [["C3", "O3"], ["O3", "HO3"]],
                     "C4": [["C4", "O4"], ["O4", "HO4"]],
                     "C5": [["C5", "C6"], ["C6", "O6"], ["O6", "HO6"]],
                     "O5": [["O5", "O5"]]}


class Atom:
    def __init__(self, atom_type, loc, charge, mass=1):
        self.atom_type = atom_type
        self.loc = loc
        self.charge = charge
        self.mass = mass
        self.neighbours = []

    def __repr__(self):
        return "<Atom {0} charge={1}, mass={2} @ {3}, {4}, {5}>"\
               .format(self.atom_type, self.charge, self.mass, *self.loc)


class Frame:
    def __init__(self, num, atom_nums=sugar_atom_nums):
        self.num = num
        self.atoms = []
        self.remark = []
        self.title = ""
        self.atom_nums = atom_nums
#       set functions to be used to calculate properties
        self.calc_measure = {"length": self.get_bond_lens,
                             "angle": self.get_bond_angles,
                             "dihedral": self.get_bond_dihedrals}

    def __repr__(self):
        return "<Frame {0} containing {1} atoms>\n{2}"\
               .format(self.num, len(self.atoms), self.title)

    def bond_length(self, num1, num2):
        dist = np.linalg.norm(self.atoms[num1].loc - self.atoms[num2].loc)
        return dist

    def bond_length_atoms(self, atom1, atom2):
        dist = sqrt((atom1.loc[0]-atom2.loc[0])**2 +
                    (atom1.loc[1]-atom2.loc[1])**2 +
                    (atom1.loc[2]-atom2.loc[2])**2)
        return dist

    def bond_angle(self, num1, num2, num3, num4):
        """
        angle at atom2 formed by bonds: 1-2 and 2-3
        """
        vec1 = (self.atoms[num2].loc - self.atoms[num1].loc)
        vec2 = (self.atoms[num4].loc - self.atoms[num3].loc)
        vec1 = vec1 / sqrt(vec1[0]**2 + vec1[1]**2 + vec1[2]**2)
        vec2 = vec2 / sqrt(vec2[0]**2 + vec2[1]**2 + vec2[2]**2)
        angle = np.arccos(np.dot(vec1, vec2))
        return 180 - (angle * 180 / np.pi)

    def norm_bisec(self, num1, num2, num3):
        """
        return normal vector to plane formed by 3 atoms
        and their bisecting vector
        """
        vec1 = (self.atoms[num2].loc - self.atoms[num1].loc)
        vec2 = (self.atoms[num3].loc - self.atoms[num2].loc)
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        normal = np.cross(vec1, vec2)
        bisec = (vec1 + vec2) / 2.
        return polar_coords(normal), polar_coords(bisec)

    def show_atoms(self, start=0, end=-1):
        """
        print coordinates of the atoms numbered start to end
        """
        print(self.title)
        if end == -1:
            end = len(self.atoms)
        for i in xrange(start, end):
            print(self.atoms[i])

    def get_bond_lens(self, request=bond_pairs):
        dists = np.zeros(len(request))
        for i, pair in enumerate(request):
            dists[i] = self.bond_length(self.atom_nums[pair[0]],
                                        self.atom_nums[pair[1]])
        return dists

    def get_bond_angles(self, request=bond_triples):
        angles = np.zeros(len(request))
#       use the same code as dihedrals, just give the same atom twice
        for i, triple in enumerate(request):
            angles[i] = self.bond_angle(self.atom_nums[triple[0]],
                                        self.atom_nums[triple[1]],
                                        self.atom_nums[triple[1]],
                                        self.atom_nums[triple[2]])
        return angles

    def get_bond_dihedrals(self, request=bond_quads):
        dihedrals = np.zeros(len(request))
        for i, quad in enumerate(request):
            dihedrals[i] = self.bond_angle(self.atom_nums[quad[0]],
                                           self.atom_nums[quad[1]],
                                           self.atom_nums[quad[2]],
                                           self.atom_nums[quad[3]])
        return dihedrals


def read_xtc_setup(grofile, xtcfile, cutoff=0, cm_map=False):
    """
    setup Frame object for reading trajectory into
    select atoms in sugar and set atom numbers
    """
    global sugar_atom_nums, verbose
    u = AtomGroup.Universe(grofile, xtcfile)
    sel = u.selectAtoms("not resname SOL")
    res_name = sel.resnames()[0]
    if verbose:
        print(sel)
        print(res_name)
        print(sel.names())
    for name in sel.names():
        sugar_atom_nums[name] = list(sel.names()).index(name)
#       if a cutoff is specified it means we want to calculate solvent RDF
    if cutoff:
        sel = u.selectAtoms("resname " + res_name +
                            " or around " + str(cutoff) +
                            " resname " + res_name)
        if verbose:
            print(sel.resnames())
            print(sel.names())
    num_frames = len(u.trajectory)
    frame = Frame(0)
    ts_init = u.trajectory[0]
    for pack in zip(sel.names(), sel.get_positions(ts_init), sel.masses()):
        atomname, coords, mass = pack
        if not cm_map:
            mass = 1
        frame.atoms.append(Atom(atomname, coords,
                                atomic_charges[atomname], mass=mass))
    cg_frame = map_cg_solvent_within_loop(0, frame)
    print(num_frames)
    print("Done xtc setup\n"+"-"*20)
#   return the total number of frames in trajectory
#   placeholders for the two Frame objects
#   and the two internal trajectory pieces
    return num_frames, frame, cg_frame, sel, u


def read_xtc_frame(sel, ts, frame_num, frame, cg_frame):
    """
    read a single frame from trajectory
    store it in the same Frame object as was used previously - efficient
    """
    global cm_map
    frame.num = frame_num
    for i, dat in enumerate(zip(sel.names(),
                                sel.get_positions(ts),
                                sel.masses())):
        atomname, coords, mass = dat[0], dat[1], dat[2]
        if not cm_map:
            mass = 1
        frame.atoms[i] = Atom(atomname, coords,
                              atomic_charges[atomname], mass=mass)
    cg_frame = map_cg_solvent_within_loop(frame_num, frame, cg_frame)
    return frame, cg_frame


def map_cg_solvent_within_loop(curr_frame, frame, cg_frame=0):
    """
    perform CG mapping using cg_map list of lists
    with current cg_map does a simple heavy atom mapping

    will be CM or GC depending on how 'Atom.mass' was set previously
        if mapping is changed in cg_map to include other atoms

    should remove the setup code into its own function (or the main xtc setup)
    """
    global cg_atom_nums
    if curr_frame == 0:
        cg_frame = Frame(curr_frame, cg_atom_nums)
    cg_frame.num = curr_frame
    for i, site in enumerate(cg_sites):
        coords = np.zeros(3)
        tot_mass = 0.
        charge = 0.
        for atom in cg_map[i]:
            mass = frame.atoms[sugar_atom_nums[atom]].mass
            tot_mass = tot_mass + mass
            coords = coords + mass*frame.atoms[sugar_atom_nums[atom]].loc
            charge = charge + frame.atoms[sugar_atom_nums[atom]].charge
        coords = coords / tot_mass  # number of atoms cancels out
        if curr_frame == 0:
            cg_frame.atoms.append(Atom(site, coords, charge))
        else:
            cg_frame.atoms[i] = Atom(site, coords, charge)
        if curr_frame == 0:
            cg_atom_nums[site] = i
    j = len(cg_sites)
    for atom in frame.atoms:
        if atom.atom_type == "OW":
            if curr_frame == 0:
                cg_frame.atoms.append(Atom("OW", atom.loc, 0.0))
            else:
                cg_frame.atoms[j] = Atom("OW", atom.loc, 0.0)
            j += 1
    return cg_frame


def solvent_rdf(cg_frame, rdf_frames=0, export=False):
    """
    calculate solvent RDF
    still needs work, doesn't look like an RDF
    """
    if rdf_frames == 0:
        rdf_frames = [[], [], [], [], [], []]
    for origin_name in cg_atom_nums:
        origin_num = cg_atom_nums[origin_name]
        origin_atom = cg_frame.atoms[origin_num]
        for far_atom in cg_frame.atoms:
            if far_atom.atom_type == "OW":
                dist = cg_frame.bond_length_atoms(origin_atom, far_atom)
                rdf_frames[origin_num].append(dist)
    return rdf_frames


def plot_rdf(rdf_frames):
    fig, ax = plt.subplots(2, 3)
    fig.tight_layout()
    for i, item in enumerate(rdf_frames):
        ax1 = plt.subplot(2, 3, i+1)
        data = ax1.hist(item, bins=100, normed=1)
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=90)
    plb.savefig("rdfs.pdf", bbox_inches="tight")
    return


def calc_measures(frame, out_file, req="length",
                  request=bond_quads, export=True):
    """
    multipurpose code to calculate bond lengths, angles, dihedrals
    output to file to allow concat between simulations
    """
    measures = []
    measures.append(frame.calc_measure[req](request))
    if export:
        measures_text = [str(num) for num in measures[-1]]
        out_file.write(",".join(measures_text) + "\n")
    return measures


def polar_coords(xyz, ax1=np.array([0, 0, 0]),
                 ax2=np.array([0, 0, 0]), mod=True):
    """
    Convert cartesian coordinates to polar, if axes given will be reoriented
    axis points to the north pole (lat), axis2 points to 0 on equator (long)
    if mod, do angles properly within -pi, +pi
    """
    tpi = 2*np.pi
    polar = np.zeros(3)
    xy = xyz[0]**2 + xyz[1]**2
    polar[0] = np.sqrt(xy + xyz[2]**2)
    polar[1] = np.arctan2(np.sqrt(xy), xyz[2]) - ax1[1]
    polar[2] = np.arctan2(xyz[1], xyz[0]) - ax2[2]
    if axis2[1] < 0:
        polar[2] = polar[2] + tpi
    if mod:
        polar[1] = polar[1] % (tpi)
        polar[2] = polar[2] % (tpi)
    return polar


def calc_dipoles(cg_frame, frame, out_file, export=True,
                 cg_internal_bonds=cg_internal_bonds,
                 sugar_atom_nums=sugar_atom_nums, adjacent=adjacent):
    """
    dipole of a charged fragment is dependent on where you measure it from
    should modify this to include OW dipoles
    """
    old_dipoles = False
    dipoles = []
    frame_dipoles = np.zeros((len(cg_sites), 3))
    for i, site in enumerate(cg_sites):
        dipole = np.zeros(3)
        if old_dipoles:     # calculate dipole from sum of bonds
            for j, bond in enumerate(cg_internal_bonds[site]):
                atom1 = frame.atoms[sugar_atom_nums[bond[0]]]
                atom2 = frame.atoms[sugar_atom_nums[bond[1]]]
                dipole += (atom1.loc - atom2.loc) *\
                          (atom1.charge - atom2.charge)
        else:   # dipole wrt origin then transpose
            for atom_name in cg_internal_map[site]:
                atom = frame.atoms[sugar_atom_nums[atom_name]]
                dipole += atom.loc * atom.charge    # first calc from origin
            cg_atom = cg_frame.atoms[cg_atom_nums[site]]
            dipole -= cg_atom.loc * cg_atom.charge  # then recentre it
        norm, bisec = cg_frame.norm_bisec(cg_atom_nums[adjacent[site][0]], i,
                                          cg_atom_nums[adjacent[site][1]])
        frame_dipoles[i] += polar_coords(dipole, norm, bisec)
        if export:
            np.savetxt(out_file, frame_dipoles, delimiter=",")
        dipoles.append(frame_dipoles)
    return dipoles


def read_energy(energyfile, export=False):
    """
    read energies from GROMACS .log file
    concat into a single file to be plot later
    """
    global verbose
    t_start = time.clock()
    print("Reading energies")
    try:
        f = open(energyfile, "r")
    except IOError:
        print("Error reading energy file")
        return []
    energy_block = False
    this_line = False
    energies_str = []
    for line in f.readlines():
        if this_line:
            energies_str.append(line.split()[1])
            this_line = False
            energy_block = False
        if energy_block:
            if not this_line and line.strip()[0:7] == "Kinetic":
                this_line = True    # the next line is the one we want
        elif line.strip()[0:8] == "Energies":
            energy_block = True
    f.close()
    if export:
        try:
            f = open("energies.csv", "a")
        except IOError:
            print("Error opening energy ouput file")
        else:
            f.write("\n".join(energies_str))
            f.close()
    energies = np.array(energies_str).astype(np.float)
    t_end = time.clock()
    print("Read {0} energies in {1}s".format(len(energies), (t_end - t_start)))
    print("-"*20)
    return energies


def print_output(output_all, output, request):
    for name, val in zip(request, output):
        print("{0}: {1:4.3f}".format("-".join(name), val))


def export_props(grofile, xtcfile, energyfile="", export=False,
                 do_dipoles=False, cutoff=0, cm_map=False):
    global verbose
    t_start = time.clock()
    pack = read_xtc_setup(grofile, xtcfile, keep_atomistic=do_dipoles,
                          cutoff=cutoff, cm_map=cm_map)
    num_frames, frame, cg_frame, sel, univ = pack
    np.set_printoptions(precision=3, suppress=True)
    rdf_frames = []
    if export:
        f_dist = open("bond_lengths.csv", "a")
        f_angle = open("bond_angles.csv", "a")
        f_dihedral = open("bond_dihedrals.csv", "a")
        f_dipole = open("dipoles.csv", "a")
    if energyfile:
        read_energy(energyfile, export=export)
    for frame_num, ts in enumerate(univ.trajectory):
        perc = frame_num * 100. / num_frames
        if(frame_num % 100 == 0):
            sys.stdout.write("\r{:2.0f}% ".format(perc) +
                             "X" * int(0.2*perc) + "-" * int(0.2*(100-perc)))
            sys.stdout.flush()
        frame, cg_frame = read_xtc_frame(sel, ts, frame_num, frame, cg_frame)
        cg_dists = calc_measures(cg_frame, f_dist,
                                 "length", cg_bond_pairs, export=export)
        cg_angles = calc_measures(cg_frame, f_angle,
                                  "angle", cg_bond_triples, export=export)
        cg_dihedrals = calc_measures(cg_frame, f_dihedral,
                                     "dihedral", cg_bond_quads, export=export)
        if do_dipoles:
            cg_dipoles = calc_dipoles(cg_frame, frame, f_dipole, export)
        if cutoff:
            if frame_num == 0:
                rdf_frames = solvent_rdf(cg_frame, 0, export=export)
            else:
                rdf_frames = solvent_rdf(cg_frame, rdf_frames, export=export)
    if export:
        f_dist.close()
        f_angle.close()
        f_dihedral.close()
        f_dipole.close()
        plot_rdf(rdf_frames)
    t_end = time.clock()
    print("\rCalculated {0} frames in {1}s\n"
          .format(num_frames, (t_end - t_start)) + "-"*20)
    return num_frames

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-g", "--gro", action="store",
                      type="string", dest="grofile",
                      help="Gromacs .gro topology", metavar="FILE")
    parser.add_option("-x", "--xtc", action="store",
                      type="string", dest="xtcfile",
                      help="Gromacs .xtc trajectory", metavar="FILE")
    parser.add_option("-E", "--energy", action="store",
                      type="string", dest="energyfile", default="",
                      help="Parse energies from log file", metavar="FILE")
    parser.add_option("-r", "--rdf", action="store",
                      type="int", dest="cutoff", default=0,
                      help="Cutoff radius for RDF calculation", metavar="INT")
    parser.add_option("-e", "--export", action="store_true",
                      dest="export", default=False,
                      help="Save data to .csv files in working directory")
    parser.add_option("-d", "--dipole", action="store_true",
                      dest="dipoles", default=False,
                      help="Calculate dipoles")
    parser.add_option("-m", "--mass", action="store_true",
                      dest="cm_map", default=False,
                      help="Use centre of mass mapping")
    parser.add_option("-v", "--verbose", action="store_true",
                      dest="verbose", default=False,
                      help="Make more verbose")
    (options, args) = parser.parse_args()
    verbose = options.verbose
    cm_map = options.cm_map
    if not options.grofile or not options.xtcfile:
        print("Must provide .gro and .xtc files to run")
        sys.exit(1)
    if verbose:
        cProfile.run("export_props(options.grofile, options.xtcfile,
                     energyfile=options.energyfile, export=options.export,
                     do_dipoles=options.dipoles, cutoff=options.cutoff,
                     cm_map=options.cm_map)", "profile")
        p = pstats.Stats("profile")
        p.sort_stats('time').print_stats(25)
    else:
        export_props(options.grofile, options.xtcfile,
                     energyfile=options.energyfile, export=options.export,
                     do_dipoles=options.dipoles, cutoff=options.cutoff,
                     cm_map=options.cm_map)
