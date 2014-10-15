#!/usr/bin/env python
import process_cython
import sys
import cProfile
import pstats

if __name__ == "__main__":
    grofile, xtcfile = sys.argv[1], sys.argv[2]
    try:
        export = int(sys.argv[3])
    except IndexError:
        export = False
    try:
        do_dipoles = int(sys.argv[4])
    except IndexError:
        do_dipoles = False
    process_cython.export_props(grofile, xtcfile, export=export, do_dipoles=do_dipoles)