import process
import post
import os
from glob import glob

def test_post_general():
    res = process.export_props("test_data/npt.gro", "test_data/npt.xtc", export=True, do_dipoles=True)
    assert post.process_all("auto") == 1
    print("cleaning")
    print(glob("*.csv"))
    for f in glob("*.csv"):
        os.remove(f)
    print(glob("*.pdf"))
    for f in glob("*.pdf"):
        os.remove(f)