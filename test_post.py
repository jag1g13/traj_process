import process
import post
import os
from glob import glob

def test_post_general():
    res = process.export_props("test_data/npt.gro", "test_data/npt.xtc", export=True, do_dipoles=True)
    res2 = post.process_all("auto")
    print("cleaning")
    print(glob("*.csv"))
    for f in glob("*.csv"):
        os.remove(f)
    print(glob("*.pdf"))
    for f in glob("*.pdf"):
        os.remove(f)
    assert res2 == 1