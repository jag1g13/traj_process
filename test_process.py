import os
import process

def test_process_all():
    #tests that the whole process completes
    res = process.export_props("test_data/npt.gro", "test_data/npt.xtc", export=True)
    assert res==151
    
def test_export_lengths():
    try:
        f1 = open("bond_lengths.csv")
        f2 = open("test_data/bond_lengths.csv")
    except IOError:
        print("Can't access bond lengths")
    else:
        match = True
        for lines in zip(f1.readlines(), f2.readlines()):
            if not lines[0] == lines[1]:
                match = False
    assert match == True
    f1.close()
    f2.close()
    os.remove("bond_lengths.csv")

def test_export_angles():
    try:
        f1 = open("bond_angles.csv")
        f2 = open("test_data/bond_angles.csv")
    except IOError:
        print("Can't access bond angles")
    else:
        match = True
        for lines in zip(f1.readlines(), f2.readlines()):
            if not lines[0] == lines[1]:
                match = False
    assert match == True
    f1.close()
    f2.close()
    os.remove("bond_angles.csv")

def test_export_dihedrals():
    try:
        f1 = open("bond_dihedrals.csv")
        f2 = open("test_data/bond_dihedrals.csv")
    except IOError:
        print("Can't access bond dihedrals")
    else:
        match = True
        for lines in zip(f1.readlines(), f2.readlines()):
            if not lines[0] == lines[1]:
                match = False
    assert match == True
    f1.close()
    f2.close()
    os.remove("bond_dihedrals.csv")
 
def test_dipole_prelim():
    #this doesn't test that the dipole code actually works, just that it doesn't crash
    res = process.export_props("test_data/npt.gro", "test_data/npt.xtc", do_dipoles=True)
    assert res==151