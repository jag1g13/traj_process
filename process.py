#from __future__ import print_function
import sys
import numpy as np
import time
import matplotlib.pyplot as plt

sugar_atom_nums = {}
#sugar_atoms = ["C4", "O4", "HO4", "C3", "O3", "HO3", "C2", "O2",\
               #"HO2", "C6", "O6", "HO6", "C5", "O5", "C1", "O1",\
               #"HO1"]
sugar_atoms = ["C1", "O1", "HO1", "C2", "O2", "HO2",\
               "C3", "O3", "HO3", "C4", "O4", "HO4",\
               "C5", "O5", "C6",  "O6", "HO6"]

#bond lengths we want to know
bond_pairs = [["C1", "O1"], ["O1", "HO1"], ["C1", "C2"],\
              ["C2", "O2"], ["O2", "HO2"], ["C2", "C3"],\
              ["C3", "O3"], ["O3", "HO3"], ["C3", "C4"],\
              ["C4", "O4"], ["O4", "HO4"], ["C4", "C5"],\
              ["C5", "O5"], ["C5", "C6"],\
              ["C6", "O6"], ["O6", "HO6"]]

#bond angles we want to know
bond_triples = []

#bond dihedrals we want to know
bond_quads = []

class Atom:
    def __init__(self, atom_type, loc):
        self.atom_type = atom_type
        self.loc = loc

class Frame:
    def __init__(self, num):
        self.num = num
        self.atoms = []
        self.remark = []
        self.title = ""
        
    def bond_length(self, num1, num2):
        """
        return Euclidean distance between two atoms
        """
        dist = np.linalg.norm(self.atoms[num1].loc - self.atoms[num2].loc)
        #print("Distance between atoms {0} and {1} is {2}".format(num1, num2, dist))
        return dist
    
    def bond_angle(self, num1, num2, num3):
        """
        angle at atom2 formed by bonds: 1-2 and 2-3
        """
        vec1 = (self.atoms[num2].loc - self.atoms[num1].loc)
        vec2 = (self.atoms[num3].loc - self.atoms[num2].loc)
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        angle = np.arccos(np.dot(vec1, vec2))
        return angle * 180 / np.pi

    def show_atoms(self, start=0, end=-1):
        """
        print coordinates of the atoms numbered start to end
        """
        print(self.title)
        if end == -1:
            end = len(self.atoms)
        for i in xrange(start, end):
            print(self.atoms[i].atom_type, self.atoms[i].loc)

    def get_all_dist_tri(self):
        size = len(sugar_atoms)
        dists = np.zeros([size, size])
        for i in xrange(size):
            for j in xrange(i):
                dists[i][j] = self.bond_length(sugar_atom_nums[sugar_atoms[i]], sugar_atom_nums[sugar_atoms[j]])
            #print(dists[i])
        dists = dists + dists.T - np.diag(dists.diagonal())
        return dists

    def get_bond_len(self):
        size = len(sugar_atoms)
        dists = np.zeros([size, size])
        for pair in bond_pairs:
            i, j = sugar_atom_nums[pair[0]], sugar_atom_nums[pair[1]]
            dists[i][j] = self.bond_length(i, j)
            #print(dists[i])
        dists = dists + dists.T - np.diag(dists.diagonal())
        return dists
    
    def get_all_angle_tri(self):
        size = len(sugar_atoms)
        angles = np.zeros([size, size, size])
        for i in xrange(size):
            for j in xrange(i):
                for k in xrange(j):
                    angles[i][j][k] = self.bond_angle(sugar_atom_nums[sugar_atoms[i]], sugar_atom_nums[sugar_atoms[j]], sugar_atom_nums[sugar_atoms[k]])
            #print(dists[i])
        angles = angles + angles.T - np.diag(angles.diagonal())
        return angles

def read(name, frame_max=float("inf")):
    """
    read frames from pdb file.  can set number of frames to read, default - all
    """
    global sugar_atom_nums
    t_start = time.clock()
    try:
        f = open(name, "r")
    except IOError:
        print("IOError when trying to open file - is the filename correct?")
        sys.exit(1)
    curr_frame = 0
    frames = [Frame(0)]
    lines = f.readlines()
    print("Reading file - length {0} lines".format(len(lines)))
    line_num = 0
    for line in lines:
        line_num += 1
        perc = line_num * 100. / len(lines)
        if(line_num%100000 == 0):                           #print progress every 100k lines
            sys.stdout.write("\r{:2.0f}% ".format(perc) + "X" * int(0.2*perc) + "-" * int(0.2*(100-perc)) )
            sys.stdout.flush()
        line = line.strip("\n")
        #print(line)
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
            #print(line)
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
            frames[curr_frame].atoms.append(Atom(atom_type, np.array(coords)))    #must be a normal line, read coords
            if curr_frame == 1:
                sugar_atom_nums[atom_type] = int(line[7:11].strip()) - 1            #which atom number is each of the sugar atoms
    frames.pop(-1)                                              #get rid of last frame, it's empty
    t_end = time.clock()
    print("\rSuccessfully read {0} frames in {1}s".format(len(frames), (t_end - t_start)))
    #print(sugar_atom_nums)
    return frames

def calc_dists(frames):
    t_start = time.clock()
    dists = []
    angles = []
    for i, frame in enumerate(frames):
        perc = i * 100. / len(frames)
        #print("Frame {0}".format(frame.num))
        if(i%100 == 0):
            sys.stdout.write("\r{:2.0f}% ".format(perc) + "X" * int(0.2*perc) + "-" * int(0.2*(100-perc)) )
            sys.stdout.flush()
        #dists.append(frame.get_all_dist_tri())
        dists.append(frame.get_bond_len())
        #angles.append(frame.get_all_angle_tri())
    t_end = time.clock()
    print("\rSuccessfully calculated {0} frames in {1}s".format(len(frames), (t_end - t_start)))
    return dists


if __name__ == "__main__":
    frames = read(r"test_cases/md.pdb")
    dists = calc_dists(frames)
    avg = np.mean(dists, axis=0)
    #bond1 = []
    #for frame in dists:
       #bond1.append(frame[0][1]) 
    #plt.hist(bond1, 100)
    #plt.show()
    np.set_printoptions(precision=3, suppress=True, linewidth=len(sugar_atoms)*7+4)
    print(avg)