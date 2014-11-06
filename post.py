#!/usr/bin/env python
from optparse import OptionParser
import cProfile
import pstats

from post_graph import *
from post_dipole import *

if __name__ == "__main__":
    global export
    parser = OptionParser()
    parser.add_option("-a", "--auto",
                      action="store_true", dest="auto", default=False,
                      help="Plot everything automatically")
    parser.add_option("-d", "--dipoles",
                      action="store_true", dest="only_3d", default=False,
                      help="Automatically do only interactive 3d dipoles")
    parser.add_option("-e", "--export",
                      action="store_true", dest="export", default=True,
                      help="Save fitting parameters")
    (options, args) = parser.parse_args()
    export = options.export
    process_all(options.auto, options.only_3d)
