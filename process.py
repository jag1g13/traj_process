import numpy
import re
import time
import cPickle
import matplotlib.pyplot as plt
from scipy.spatial import distance

sugar_atom_nums = {}
sugar_atoms = ["C4", "O4", "HO4", "C3", "O3", "HO3", "C2", "O2", "HO2", "C6",\
            "O6", "HO6", "C5", "O5", "C1", "O1", "HO1"]

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
        
    def atom_dist(self, num1, num2):
        """
        return Euclidean distance between two atoms
        """
        #print("Distance between atoms {0} and {1}".format(num1, num2))
        dist = distance.euclidean(self.atoms[num1].loc, self.atoms[num2].loc)
        #print(dist)
        return dist

    def show_atoms(self, start=0, end=-1):
        """
        print coordinates of the atoms numbered start to end
        """
        print(self.title)
        for i in xrange(start, end):
            print(self.atoms[i].atom_type, self.atoms[i].loc)

	

def read(name, frame_max=float("inf")):
    """
    read frames from pdb file.  can set number of frames to read, default - all
    """
    global sugar_atom_nums
    f = open(name, "r")
    if not f:
        print("Error opening file")
    curr_frame = 0
    frames = [Frame(0)]
    coord_re2 = re.compile(r"\d{1,2}\.\d{3}")               #read all floats in the right format
    type_re = re.compile(r"[A-Z]{1,2}\d")
    lines = f.readlines()
    print("Reading file - length {0} lines".format(len(lines)))
    line_num = 0
    for line in lines:
        line_num += 1
        perc = line_num * 100. / len(lines)
        if(line_num%100000 == 0):                           #print progress every 100k lines
            print("{0}% done".format(perc))
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
            if not (int(line.split()[1]) == curr_frame+1):
                print("Frame number incorrect")
            continue
        coords = [float(num) for num in re.findall(coord_re2, line)]
        #print(line)
        #atom_type = re.search(type_re, line).group()
        atom_type = line[12:17].strip()
        #print(coords)
        #frames[curr_frame].atoms.append(numpy.array(coords))    #must be a normal line, read coords
        frames[curr_frame].atoms.append(Atom(atom_type, numpy.array(coords)))    #must be a normal line, read coords
        if curr_frame == 1:
            sugar_atom_nums[atom_type] = int(line[7:11].strip())
    frames.pop(-1)                                              #get rid of last frame, it's empty
    del(sugar_atom_nums["HW1"])                                 #remove entries for water from dictionary, don't need them
    del(sugar_atom_nums["HW2"])
    del(sugar_atom_nums["OW"])
    print("Read {0} frames".format(len(frames)))
    print(sugar_atom_nums)
    return frames

def get_all_dist(frame):
    dists = []
    for atom1 in sugar_atoms:
        dists.append([])
        for atom2 in sugar_atoms:
            dists[-1].append(frame.atom_dist(sugar_atom_nums[atom1], sugar_atom_nums[atom2]))
        #print(dists[-1])
    return dists

if __name__ == "__main__":
    start_frame = time.clock()
    frames = read(r"test_cases/md.pdb")
    #frames = cPickle.load(open("frames.pickle", "rb"))
    end_frame =  time.clock()
    print(end_frame - start_frame)
    #start_pickle = time.clock()
    #cPickle.dump(frames, open("frames.pickle", "wb"))
    #end_pickle = time.clock()
    #print(end_pickle - start_pickle)
    #print(frames[1].atoms[0:20])
    frames[0].show_atoms(0,20)
    #print(frames[-1].atoms[0:20])
    frames[-1].show_atoms(0,20)
    #frames[0].atom_dist(0,1)
    dist = []
    start_dist = time.clock()
    for frame in frames:
        #dist.append(frame.atom_dist(0,1))
        #print("Frame {0}".format(frame.num))
        dists = get_all_dist(frame)
    for line in dists:
        print(line)
    end_dist = time.clock()
    #print(numpy.average(dist))
    #plt.hist(dist, 100)
    #plt.show()
    print(end_dist - start_dist)