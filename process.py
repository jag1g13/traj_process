#!/usr/bin/env python
#from __future__ import print_function
import sys
import numpy as np
import time
import os.path
from subprocess import call
import matplotlib.pyplot as plt
#from libxdrfile2 import xdrfile_open, xdrfile_close, read_xtc_natoms, read_xtc, DIM, exdrOK
import MDAnalysis.coordinates.xdrfile.libxdrfile2 as xdr
import MDAnalysis.core.AtomGroup as AtomGroup
from scipy import optimize
import pylab as plb

sugar_atom_nums = {}
#sugar_atoms = ["C4", "O4", "HO4", "C3", "O3", "HO3", "C2", "O2",\
               #"HO2", "C6", "O6", "HO6", "C5", "O5", "C1", "O1",\
               #"HO1"]
sugar_atoms = ["C1", "O1", "HO1", "C2", "O2", "HO2",\
               "C3", "O3", "HO3", "C4", "O4", "HO4",\
               "C5", "O5", "C6",  "O6", "HO6"]

#atomic charges
atomic_charges = {"C4": 0.232, "O4": -0.642, "HO4": 0.410, "C3": 0.232, "O3": -0.642, "HO3": 0.410, "C2": 0.232, "O2": -0.642,\
                  "HO2": 0.410, "C6": 0.232, "O6": -0.642, "HO6": 0.410, "C5": 0.376, "O5": -0.480, "C1": 0.232, "O1": -0.538,\
                  "HO1": 0.410}
               
#bond lengths we want to know
bond_pairs = [["C1", "O1"], ["O1", "HO1"], ["C1", "C2"],\
              ["C2", "O2"], ["O2", "HO2"], ["C2", "C3"],\
              ["C3", "O3"], ["O3", "HO3"], ["C3", "C4"],\
              ["C4", "O4"], ["O4", "HO4"], ["C4", "C5"],\
              ["C5", "O5"], ["C5", "C6"],\
              ["C6", "O6"], ["O6", "HO6"]]

#bond angles we want to know
bond_triples = [["O1", "C1", "C2"], ["C1", "O1", "HO1"], ["C1", "C2", "O2"], ["C1", "C2", "C3"],\
                ["O2", "C2", "C3"], ["C2", "O2", "HO2"], ["C2", "C3", "O3"], ["C2", "C3", "C4"],\
                ["O3", "C3", "C4"], ["C3", "O3", "HO3"], ["C3", "C4", "O4"], ["C3", "C4", "C5"],\
                ["O4", "C4", "C5"], ["C4", "O4", "HO4"], ["C4", "C5", "C6"], ["C4", "C5", "O5"],\
                ["C6", "O5", "C1"], ["C5", "C6", "HO6"], ["C5", "O5", "C1"],\
                ["O5", "C1", "O1"], ["O5", "C1", "C2"]]

#bond dihedrals we want to know
bond_quads = []

#names of the cg sites
cg_sites = ["C1", "C2", "C3", "C4", "C5", "O5"]

#atom numbers of the cg sites
cg_atom_nums = {}

#which atoms map to which cg site, same order as above
#cg_map = [["C1", "O1", "HO1"], ["C2", "O2", "HO2"], ["C3", "O3", "HO3"],\
          #["C4", "O4", "HO4"], ["C5", "C6", "O6", "HO6"], ["O5"]]
#cg_map = [["C1", "O1"], ["C2", "O2"], ["C3", "O3"],\
          #["C4", "O4"], ["C5", "C6"], ["O5"]]
cg_map = [["C1"], ["C2"], ["C3"],\
          ["C4"], ["C5"], ["O5"]]

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

#bonds within a cg site
cg_internal_bonds = {"C1": [["C1", "O1"], ["O1", "HO1"]],\
                     "C2": [["C2", "O2"], ["O2", "HO2"]],\
                     "C3": [["C3", "O3"], ["O3", "HO3"]],\
                     "C4": [["C4", "O4"], ["O4", "HO4"]],\
                     "C5": [["C5", "C6"], ["C6", "O6"], ["O6", "HO6"]],\
                     "O5": [["O5", "O5"]]}

class Atom:
    def __init__(self, atom_type, loc, charge):
        self.atom_type = atom_type
        self.loc = loc
        self.charge = charge

    def __repr__(self):
        return "<Atom {0} charge={1} @ {2}, {3}, {4}>".format(self.atom_type, self.charge, *self.loc)

class Frame:
    def __init__(self, num, atom_nums=sugar_atom_nums):
        self.num = num
        self.atoms = []
        self.remark = []
        self.title = ""
        self.atom_nums = atom_nums

    def __repr__(self):
        return "<Frame {0} containing {1} atoms>\n{2}".format(self.num, len(self.atoms), self.title)
        
    def bond_length(self, num1, num2):
        """
        return Euclidean distance between two atoms
        """
        dist = np.linalg.norm(self.atoms[num1].loc - self.atoms[num2].loc)
        #print("Distance between atoms {0} and {1} is {2}".format(num1, num2, dist))
        return dist
    
    def bond_angle(self, num1, num2, num3, num4):
        """
        angle at atom2 formed by bonds: 1-2 and 2-3
        """
        vec1 = (self.atoms[num2].loc - self.atoms[num1].loc)
        vec2 = (self.atoms[num4].loc - self.atoms[num3].loc)
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        angle = np.arccos(np.dot(vec1, vec2))
        return 180 - (angle * 180 / np.pi)
    
    def angle_norm_bisect(self, num1, num2, num3):
        """
        return normal vector to plane formed by 3 atoms and their bisecting vector
        """ 
        vec1 = (self.atoms[num2].loc - self.atoms[num1].loc)
        vec2 = (self.atoms[num3].loc - self.atoms[num2].loc)
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        normal = np.cross(vec1, vec2)
        bisec = (vec1 + vec2) / 2.
        return normal, bisec
        

    def show_atoms(self, start=0, end=-1):
        """
        print coordinates of the atoms numbered start to end
        """
        print(self.title)
        if end == -1:
            end = len(self.atoms)
        for i in xrange(start, end):
            print(self.atoms[i].atom_type, self.atoms[i].loc)

    def get_bond_lens(self, request=bond_pairs):
        dists = np.zeros(len(request))
        for i, pair in enumerate(request):
            #print(i, pair)
            dists[i] = self.bond_length(self.atom_nums[pair[0]], self.atom_nums[pair[1]])
            #print(dists[i])
        return dists

    def get_bond_angles(self, request=bond_triples):
        angles = np.zeros(len(request))
        for i, triple in enumerate(request):
            angles[i] = self.bond_angle(self.atom_nums[triple[0]], self.atom_nums[triple[1]], self.atom_nums[triple[1]], self.atom_nums[triple[2]])
        return angles

    def get_bond_dihedrals(self, request=bond_quads):
        dihedrals = np.zeros(len(request))
        for i, quad in enumerate(request):
            dihedrals[i] = self.bond_angle(self.atom_nums[quad[0]], self.atom_nums[quad[1]], self.atom_nums[quad[2]], self.atom_nums[quad[3]])
        return dihedrals

def read(filename, frame_max=float("inf")):
    """
    read frames from pdb file.  can set number of frames to read, default - all
    """
    global sugar_atom_nums
    t_start = time.clock()
    try:
        f = open(filename, "r")
        filesize = os.path.getsize(filename)
        print("Reading file - size {0:3.2f} GB".format(float(filesize)/(1024*1024*1024)))
    except IOError:
        print("IOError when trying to open file {0} - is the filename correct?".format(filename))
        sys.exit(1)
    curr_frame = 0
    frames = [Frame(0)]
    line_num = 0
    for line in f:
        line_num += 1
        perc = f.tell() * 100. / filesize
        if(line_num%100000 == 0):                           #print progress every 100k lines
            sys.stdout.write("\r{:2.0f}% ".format(perc) + "X" * int(0.2*perc) + "-" * int(0.2*(100-perc)) )
            sys.stdout.flush()
        line = line.strip("\n")
        if line == "ENDMDL":                                #end of frame
            curr_frame += 1
            if curr_frame > frame_max:
                break
            frames.append(Frame(curr_frame))
            continue
        if line.startswith("REMARK"):                       #pdb comment
            frames[curr_frame].remark.append(line)
            continue
        if line.startswith("TITLE"):                        #frame title
            frames[curr_frame].title = line
            continue
        if line == "TER":                                   #terminate frame, same as ENDMDL
            continue
        if line.startswith("CRYST"):                        #crystal parameters
            continue
        if line.startswith("MODEL"):                        #start of a frame, check frame number
            if not (int(line.split()[1]) == curr_frame+1):  #should create a new frame here, but why change something that works?
                print("Frame number incorrect")
            continue
        coords = [float(num) for num in [line[31:38], line[39:46], line[47:54]]]    #fixed format, just read off the right columns
        atom_type = line[12:17].strip()
        if atom_type in sugar_atoms:
            frames[curr_frame].atoms.append(Atom(atom_type, np.array(coords), atomic_charges[atom_type]))    #must be a normal line, read coords
            if curr_frame == 1:
                sugar_atom_nums[atom_type] = int(line[7:11].strip()) - 1            #which atom number is each of the sugar atoms
    frames.pop(-1)                                              #get rid of last frame, it's empty
    f.close()
    t_end = time.clock()
    print("\rRead {0} frames in {1}s\n".format(len(frames), (t_end - t_start)) + "-"*20)
    return frames


def read_xtc(grofile, xtcfile):
    global sugar_atom_nums
    t_start = time.clock()
    #xtc = xdr.xdrfile_open(xtcfile, 'r')
    #assert xtc != None, "Failed to open xtc"
    u = AtomGroup.Universe(grofile, xtcfile)
    sel = u.selectAtoms("not resname SOL")
    print(sel)
    print(sel.resnames())
    print(sel.names())
    for name in sel.names():
        sugar_atom_nums[name] = list(sel.names()).index(name)
    frames = []
    i = 0
    num_frames = len(u.trajectory)
    print(num_frames)
    for ts in u.trajectory:
        perc = i * 100. / num_frames
        if(i%100 == 0):
            sys.stdout.write("\r{:2.0f}% ".format(perc) + "X" * int(0.2*perc) + "-" * int(0.2*(100-perc)) )
            sys.stdout.flush()
        frames.append(Frame(i))
        for atomname, coords in zip(sel.names(), sel.get_positions(ts)):
            frames[i].atoms.append(Atom(atomname, coords, atomic_charges[atomname]))
        i += 1
    t_end = time.clock()
    print("\rRead {0} frames in {1}s\n".format(len(frames), (t_end - t_start)) + "-"*20)
    return frames


def calc_dists(frames, request=bond_pairs, export=True):
    print("Calculating bond lengths")
    if export:
        #if os.path.isfile("bond_lengths.csv"):
            #print("Bond length output already exists.")
            #filename = raw_input("Please input new name: ")
            #if not filename:
                #filename = "bond_lengths.csv"
            #f = open(filename, "a")
        #else:
        f = open("bond_lengths.csv", "a")
    t_start = time.clock()
    dists = []
    if export:
        bond_names = ["-".join(pair) for pair in request]
        f.write(",".join(bond_names) + "\n")
    for i, frame in enumerate(frames):
        perc = i * 100. / len(frames)
        #print("Frame {0}".format(frame.num))
        if(i%100 == 0):
            sys.stdout.write("\r{:2.0f}% ".format(perc) + "X" * int(0.2*perc) + "-" * int(0.2*(100-perc)) )
            sys.stdout.flush()
        dists.append(frame.get_bond_lens(request))
        if export:
            dists_text = [str(num) for num in dists[-1]]
            f.write(",".join(dists_text) + "\n")
    avg = np.mean(dists, axis=0)
    t_end = time.clock()
    if export:
        f.truncate()
        f.close()
    print("\rCalculated {0} frames in {1}s\n".format(len(frames), (t_end - t_start)) + "-"*20)
    return dists, avg


def calc_angles(frames, request=bond_triples, export=True):
    print("Calculating bond angles")
    if export:
        #if os.path.isfile("bond_angles.csv"):
            #print("Bond angle output already exists.")
            #filename = raw_input("Please input new name: ")
            #if not filename:
                #filename = "bond_angles.csv"
            #f = open(filename, "a")
        #else:
        f = open("bond_angles.csv", "a")
    t_start = time.clock()
    angles = []
    if export:
        angle_names = ["-".join(triple) for triple in request]
        f.write(",".join(angle_names) + "\n")
    for i, frame in enumerate(frames):
        perc = i * 100. / len(frames)
        #print("Frame {0}".format(frame.num))
        if(i%100 == 0):
            sys.stdout.write("\r{:2.0f}% ".format(perc) + "X" * int(0.2*perc) + "-" * int(0.2*(100-perc)) )
            sys.stdout.flush()
        angles.append(frame.get_bond_angles(request))
        if export:
            angles_text = [str(num) for num in angles[-1]]
            f.write(",".join(angles_text) + "\n")
    avg = np.mean(angles, axis=0)
    t_end = time.clock()
    if export:
        f.truncate()
        f.close()
    print("\rCalculated {0} frames in {1}s\n".format(len(frames), (t_end - t_start)) + "-"*20)
    return angles, avg


def calc_dihedrals(frames, request=bond_quads, export=True):
    print("Calculating bond dihedrals")
    if export:
        #if os.path.isfile("bond_dihedrals.csv"):
            #print("Bond dihedral output already exists.")
            #filename = raw_input("Please input new name: ")
            #if not filename:
                #filename = "bond_dihedrals.csv"
            #f = open(filename, "a")
        #else:
        f = open("bond_dihedrals.csv", "a")
    t_start = time.clock()
    dihedrals = []
    if export:
        dihedral_names = ["-".join(quad) for quad in request]
        f.write(",".join(dihedral_names) + "\n")
    for i, frame in enumerate(frames):
        perc = i * 100. / len(frames)
        #print("Frame {0}".format(frame.num))
        if(i%100 == 0):
            sys.stdout.write("\r{:2.0f}% ".format(perc) + "X" * int(0.2*perc) + "-" * int(0.2*(100-perc)) )
            sys.stdout.flush()
        dihedrals.append(frame.get_bond_dihedrals(request))
        if export:
            dihedrals_text = [str(num) for num in dihedrals[-1]]
            f.write(",".join(dihedrals_text) + "\n")
    avg = np.mean(dihedrals, axis=0)
    t_end = time.clock()
    if export:
        f.truncate()
        f.close()
    print("\rCalculated {0} frames in {1}s\n".format(len(frames), (t_end - t_start)) + "-"*20)
    return dihedrals, avg

def calc_measurements(frames, request=bond_quads, export=True):
    print("Calculating bond dihedrals")
    if export:
        if os.path.isfile("bond_dihedrals.csv"):
            print("Bond dihedral output already exists.")
            filename = raw_input("Please input new name: ")
            if not filename:
                filename = "bond_dihedrals.csv"
            f = open(filename, "a")
        else:
            f = open("bond_dihedrals.csv", "a")
    t_start = time.clock()
    dihedrals = []
    if export:
        dihedral_names = ["-".join(quad) for quad in request]
        f.write(",".join(dihedral_names) + "\n")
    for i, frame in enumerate(frames):
        perc = i * 100. / len(frames)
        #print("Frame {0}".format(frame.num))
        if(i%100 == 0):
            sys.stdout.write("\r{:2.0f}% ".format(perc) + "X" * int(0.2*perc) + "-" * int(0.2*(100-perc)) )
            sys.stdout.flush()
        dihedrals.append(frame.get_bond_dihedrals(request))
        if export:
            dihedrals_text = [str(num) for num in dihedrals[-1]]
            f.write(",".join(dihedrals_text) + "\n")
    avg = np.mean(dihedrals, axis=0)
    t_end = time.clock()
    if export:
        f.truncate()
        f.close()
    print("\rCalculated {0} frames in {1}s\n".format(len(frames), (t_end - t_start)) + "-"*20)
    return dihedrals, 

def map_cg(frames):
    global cg_atom_nums
    print("Calculating cg sites")
    t_start = time.clock()
    cg_frames = []
    for curr_frame, frame in enumerate(frames):
        perc = curr_frame * 100. / len(frames)
        #print("Frame {0}".format(frame.num))
        if(curr_frame%100 == 0):
            sys.stdout.write("\r{:2.0f}% ".format(perc) + "X" * int(0.2*perc) + "-" * int(0.2*(100-perc)) )
            sys.stdout.flush()
        cg_frames.append(Frame(curr_frame, cg_atom_nums))
        for i, site in enumerate(cg_sites):
            coords = np.zeros(3)
            charge = 0.
            for atom in cg_map[i]:
                coords = coords + frame.atoms[sugar_atom_nums[atom]].loc
                charge = charge + frame.atoms[sugar_atom_nums[atom]].charge
            coords = coords / len(cg_map[i])
            cg_frames[curr_frame].atoms.append(Atom(site, coords, charge))
            if curr_frame == 1:
                cg_atom_nums[site] = i
    t_end = time.clock()
    print("\rCalculated {0} frames in {1}s\n".format(len(frames), (t_end - t_start)) + "-"*20)
    return cg_frames


def polar_coords(xyz, axis=np.array([0,0,0]), axis2=np.array([0,0,0]), mod=0):
    """
    Convert cartesian coordinates to polar, if axes are given it will be reoriented.
    axis points to the north pole (latitude), axis2 points to 0 on equator (longitude)
    if mod, do angles properly within -pi, +pi
    """
    polar = np.zeros(3)
    xy = xyz[0]**2 + xyz[1]**2
    polar[0] = np.sqrt(xy + xyz[2]**2)
    polar[1] = np.arctan2(np.sqrt(xy), xyz[2]) - axis[1]
    polar[2] = np.arctan2(xyz[1], xyz[0]) - axis2[2]
    if mod:
        polar[1] = (polar[1]%2*np.pi)-np.pi
        polar[2] = (polar[2]%2*np.pi)-np.pi
    return polar

def calc_dipoles(cg_frames, frames, export=True):
    print("Calculating dipoles")
    t_start = time.clock()
    if export:
        f = open("dipoles.csv", "a")
    dipoles = []
    for curr_frame, cg_frame in enumerate(cg_frames):
        perc = curr_frame * 100. / len(cg_frames)
        if(curr_frame%100 == 0):
            sys.stdout.write("\r{:2.0f}% ".format(perc) + "X" * int(0.2*perc) + "-" * int(0.2*(100-perc)) )
            sys.stdout.flush()
        frame_dipoles = np.zeros((len(cg_sites),3))
        for i, site in enumerate(cg_frame.atoms):
            dipole = np.zeros(3)
            for j, bond in enumerate(cg_internal_bonds[site.atom_type]):
                atom1 = frames[curr_frame].atoms[sugar_atom_nums[bond[0]]]
                atom2 = frames[curr_frame].atoms[sugar_atom_nums[bond[1]]]
                dipole += (atom1.loc - atom2.loc) * (atom1.charge - atom2.charge)
            norm, bisec = frames[curr_frame].angle_norm_bisect((i-1)%6, i, (i+1)%6)
            frame_dipoles[i] += polar_coords(dipole, norm, bisec)
        if export:
            #f.write(*frame_dipoles)
            np.savetxt(f, frame_dipoles, delimiter=",")
        dipoles.append(frame_dipoles)
    if export:
        f.close()
    t_end = time.clock()
    print("\rCalculated {0} frames in {1}s\n".format(len(frames), (t_end - t_start)) + "-"*20)
    return dipoles

def print_output(output_all, output, request):
    for name, val in zip(request, output):
        print("{0}: {1:4.3f}".format("-".join(name), val))


def graph_output(output_all, request):
    rearrange = zip(*output_all)
    plt.figure()
    for i, item in enumerate(rearrange):
        plt.subplot(2,3, i+1)
        data = plt.hist(item, bins=100, normed=1)
        def f(x, a, b, c):
            return a * plb.exp(-(x-b)**2.0 / (2*c**2))
        x = [0.5*(data[1][j] + data[1][j+1]) for j in xrange(len(data[1])-1)]
        y = data[0]
        try:
            popt, pcov = optimize.curve_fit(f, x, y)
            x_fit = plb.linspace(x[0], x[-1], 100)
            y_fit = f(x_fit, *popt)
            plt.plot(x_fit, y_fit, lw=4, color="r")
            print(popt, pcov)
        except RuntimeError:
            print("Failed to optimise fit")

if __name__ == "__main__":
    t_start = time.clock()
    #frames = read(r"test_cases/md.pdb")
    try:
        export = int(sys.argv[3])
    except IndexError:
        export = False
    frames = read_xtc(sys.argv[1], sys.argv[2])
    #frames = read(sys.argv[1])
    np.set_printoptions(precision=3, suppress=True)
    #dists = calc_dists(frames, export=export)[1]
    #angles = calc_angles(frames, export=export)[1]
    #print("Average bond lengths")
    #for bond, dist in zip(bond_pairs, dists):
        #print("{0}-{1:3}: {2:4.3f}".format(bond[0], bond[1], dist))
    #print("Average bond angles")
    #for bond, angle in zip(bond_triples, angles):
        #print("{0}-{1}-{2:3}: {3:5.2f}".format(bond[0], bond[1], bond[2], angle))
    cg_frames = map_cg(frames)
    cg_all_dists, cg_dists = calc_dists(cg_frames, cg_bond_pairs, export=export)
    cg_all_angles, cg_angles = calc_angles(cg_frames, cg_bond_triples, export=export)
    cg_all_dihedrals, cg_dihedrals = calc_dihedrals(cg_frames, cg_bond_quads, export=export)
    cg_dipoles = calc_dipoles(cg_frames, frames, export)
    #print("Average cg bond lengths")
    #for bond, dist in zip(cg_bond_pairs, cg_dists):
        #print("{0}-{1:3}: {2:4.3f}".format(bond[0], bond[1], dist))
    print_output(cg_all_dists, cg_dists, cg_bond_pairs)
    print_output(cg_all_angles, cg_angles, cg_bond_triples)
    print_output(cg_all_dihedrals, cg_dihedrals, cg_bond_quads)
    #graph_output(cg_all_dists, cg_bond_pairs)
    #graph_output(cg_all_angles, cg_bond_triples)
    #graph_output(cg_all_dihedrals, cg_bond_quads)
    t_end = time.clock()
    print("\rCalculated {0} frames in {1}s\n".format(len(frames), (t_end - t_start)) + "-"*20)
    #plt.show()