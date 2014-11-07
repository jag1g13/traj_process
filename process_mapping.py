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

