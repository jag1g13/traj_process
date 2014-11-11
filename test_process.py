import pytest
import os
import process
from process import Atom, Frame
import numpy as np


def test_process_all():
    """
    just tests that the whole thing finishes
    """
    res = process.export_props("test_data/npt.gro",
                               "test_data/npt.xtc",
                               export=True, do_dipoles=True)
    assert res == 151


def test_export_lengths():
    """
    just check they're the same as calculated before
    """
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
    assert match
    f1.close()
    f2.close()
    os.remove("bond_lengths.csv")


def test_export_angles():
    """
    just check they're the same as calculated before
    """
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
    assert match
    f1.close()
    f2.close()
    os.remove("bond_angles.csv")


def test_export_dihedrals():
    """
    just check they're the same as calculated before
    """
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
    assert match
    f1.close()
    f2.close()
    os.remove("bond_dihedrals.csv")


def test_polar_coords_1():
    """
    test the polar coordinate conversion without axis reorientation
    """
    test = np.array([1., 0., 1.])
    res = np.array([np.sqrt(2), np.pi/4, 0.])
    assert np.array_equal(process.polar_coords(test), res)


def test_polar_coords_2():
    """
    test the polar coordinate conversion with axis reorientation
    [1,0,0] is [1, pi/2, 0]
    [0,1,0] is [1, pi/2, pi/2]
    [0,0,1] is [1, 0, 0]
    doesn't actually need reorienting here
    """
    test = np.array([0., 1., 0.])
    ax_1 = np.array([1., 0., 0.])
    ax_2 = np.array([1., np.pi/2., 0.])
    res = np.array([1., np.pi/2, np.pi/2])
    assert np.array_equal(process.polar_coords(
        test, axis1=ax_1, axis2=ax_2), res)


def test_polar_coords_3():
    """
    test the polar coordinate conversion with axis reorientation
    [1,0,0] is [1, pi/2, 0]
    [0,1,0] is [1, pi/2, pi/2]
    [0,0,1] is [1, 0, 0] ie up
    this test has an axis inverted
    """
    test = np.array([0., 1., 0.])
    ax_1 = np.array([1., 0., 0.])
    ax_2 = np.array([1., -np.pi/2., 0.])
    res = np.array([1., np.pi/2, np.pi/2])
    assert np.array_equal(process.polar_coords(
        test, axis1=ax_1, axis2=ax_2), res)


def test_polar_coords_4():
    """
    test the polar coordinate conversion with axis reorientation
    [1,0,0] is [1, pi/2, 0]
    [0,1,0] is [1, pi/2, pi/2]
    [0,0,1] is [1, 0, 0] ie up
    this test has axes displaced and not perpendicular
    """
    test = np.array([0., 1., 0.])
    ax_1 = np.array([1., np.pi/4, np.pi/2])
    ax_2 = np.array([1., np.pi/2, np.pi/4])
    res = np.array([1., np.pi/4, np.pi/4])
    assert np.array_equal(process.polar_coords(
        test, axis1=ax_1, axis2=ax_2), res)


@pytest.mark.xfail(reason="Dipole code keeps changing won't be consistent")
def test_export_dipoles():
    """
    check the dipole code doesn't crash
    check they're the same as calculated before
    """
    try:
        os.remove("dipoles.csv")
    except OSError:
        pass    # it wasn't there, nevermind
    res = process.export_props("test_data/npt.gro",
                               "test_data/npt.xtc",
                               do_dipoles=True, export=True)
    assert res == 151
    try:
        f1 = open("dipoles.csv")
        f2 = open("test_data/dipoles.csv")
    except IOError:
        print("Can't access dipoles")
        match = False
    else:
        match = True
        for lines in zip(f1.readlines(), f2.readlines()):
            if not lines[0] == lines[1]:
                print(lines)
                match = False
    assert match
    f1.close()
    f2.close()


def test_dipole_self_consistency():
    try:
        os.remove("dipoles.csv")
    except OSError:
        pass    # it wasn't there nevermind
    res = process.export_props("test_data/npt.gro",
                               "test_data/npt.xtc",
                               export=True, do_dipoles=True)
    res = process.export_props("test_data/npt.gro",
                               "test_data/npt.xtc",
                               export=True, do_dipoles=True)
    try:
        f1 = open("dipoles.csv")
    except IOError:
        print("Can't access dipoles")
    else:
        lines = f1.readlines()
        new_lines = []
        match = True
        diff = 0
        for n in xrange(906):
            if not lines[n] == lines[n+906]:
                diff = diff+1
                print(lines[n], lines[n+906])
                match = False
    assert match
    f1.close()


@pytest.mark.xfail(reason="Can't find dipole of non-sugar yet")
def test_dipole_water():
    """
    create a toy water-like molecule and calculate its dipole
    yeah, this isn't going to work any time soon
    the code doesn't support arbitrary molecules (YET)
    """
    frames = [Frame(0)]
#   frame.append(atom_type, coords, charge)
    frames[0].atoms.append(Atom("H1", np.array([1., 0., 0.]), 1.))
    frames[0].atoms.append(Atom("H2", np.array([0., 1., 0.]), 1.))
    frames[0].atoms.append(Atom("O1", np.array([0., 0., 0.]), -1.))
    internal_bonds = {"W1": [["O1", "H1"], ["O1", "H2"]]}
    atom_nums = {"H1": 0, "H2": 1, "O1": 2}
    adjacent = {"W1": ["W1", "W1"]}
    cg_frames = [Frame(0)]
    cg_frames[0].atoms.append(Atom("W1", np.array([0., 0., 0.]), 1.))
    res = process.calc_dipoles(cg_frames, frames,
                               cg_internal_bonds=internal_bonds,
                               sugar_atom_nums=atom_nums, adjacent=adjacent)


def test_clean():
    expected_files = ["bond_angles.csv",
                      "bond_lengths.csv",
                      "bond_dihedrals.csv",
                      "dipoles.csv"]
    for f in expected_files:
        found = True
        try:
            os.remove(f)
        except OSError:
            found = False
        assert found
