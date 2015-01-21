from math import sqrt
import numpy as np
from process_mapping import *

sugar_atom_nums = {}


class Atom:
    """
    Atom class holds atomic variables: atom type, position, charge and mass.
    """
    def __init__(self, atom_type, loc, charge, mass=1):
        self.atom_type = atom_type
        self.loc = loc
        self.charge = charge
        self.mass = mass
        self.neighbours = []

    def __repr__(self):
        return "<Atom {0} charge={1}, mass={2} @ {3}, {4}, {5}>" \
            .format(self.atom_type, self.charge, self.mass, *self.loc)


class Frame:
    """
    Frame class holds a list of atoms and handles operations on them.
    """
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
        return "<Frame {0} containing {1} atoms>\n{2}" \
            .format(self.num, len(self.atoms), self.title)

    def bond_length(self, num1, num2):
        """
        Returns bond length between two atoms given by their order in the frame (integers)
        """
        dist = np.linalg.norm(self.atoms[num1].loc - self.atoms[num2].loc)
        return dist

    def bond_length_atoms(self, atom1, atom2):
        """
        Returns bond length between two atoms given as Atom objects
        """
        dist = sqrt((atom1.loc[0]-atom2.loc[0])**2 +
                    (atom1.loc[1]-atom2.loc[1])**2 +
                    (atom1.loc[2]-atom2.loc[2])**2)
        return dist

    def bond_angle(self, num1, num2, num3, num4):
        """
        Returns the bond angle between the vectors atom2-atom1 and atom4-atom3
        Coresponds to a dihedral if atoms 2 and 3 are different, bond angle if the same
        """
        vec1 = (self.atoms[num2].loc - self.atoms[num1].loc)
        vec2 = (self.atoms[num4].loc - self.atoms[num3].loc)
        vec1 /= sqrt(vec1[0]**2 + vec1[1]**2 + vec1[2]**2)
        vec2 /= sqrt(vec2[0]**2 + vec2[1]**2 + vec2[2]**2)
        angle = np.arccos(np.dot(vec1, vec2))
        return 180 - (angle * 180 / np.pi)

    def norm_bisec(self, num1, num2, num3):
        """
        Return normal vector to plane formed by 3 atoms
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
        Print coordinates of the atoms numbered start to end
        """
        print(self.title)
        if end == -1:
            end = len(self.atoms)
        for i in xrange(start, end):
            print(self.atoms[i])

    def get_bond_lens(self, request=cg_bond_pairs):
        """
        Calculates and returns all bond lengths requested in the passed list
        """
        dists = np.zeros(len(request))
        for i, pair in enumerate(request):
            dists[i] = self.bond_length(self.atom_nums[pair[0]],
                                        self.atom_nums[pair[1]])
        return dists

    def get_bond_angles(self, request=cg_bond_triples):
        """
        Calculates and returns all bond angles requested in the passed list
        """
        angles = np.zeros(len(request))
        #       use the same code as dihedrals, just give the same atom twice
        for i, triple in enumerate(request):
            angles[i] = self.bond_angle(self.atom_nums[triple[0]],
                                        self.atom_nums[triple[1]],
                                        self.atom_nums[triple[1]],
                                        self.atom_nums[triple[2]])
        return angles

    def get_bond_dihedrals(self, request=cg_bond_quads):
        """
        Calculates and returns all bond dihedrals requested in the passed list
        """
        dihedrals = np.zeros(len(request))
        for i, quad in enumerate(request):
            dihedrals[i] = self.bond_angle(self.atom_nums[quad[0]],
                                           self.atom_nums[quad[1]],
                                           self.atom_nums[quad[2]],
                                           self.atom_nums[quad[3]])
        return dihedrals
