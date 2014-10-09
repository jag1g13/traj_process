import process

def test_process_all():
    res = process.export_props("test_data/npt.gro", "test_data/npt.xtc")
    assert res==151

def test_dipole_1():
    res = process.export_props("test_data/npt.gro", "test_data/npt.xtc", do_dipoles=True)
    assert res==151