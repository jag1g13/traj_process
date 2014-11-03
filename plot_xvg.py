#!/usr/bin/env python
import matplotlib.pyplot as plt
from optparse import OptionParser
import sys


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", "--file",
                      action="store", type="string", dest="xvgfile",
                      help="xvg file to plot", metavar="FILE")
    (options, args) = parser.parse_args()
    try:
        f = open(options.xvgfile, "r")
    except IOError:
        print("Error opening file")
        sys.exit()
    xs = []
    ys = []
    plt.figure()
    for line in f.readlines():
        if line[0] == "#":
            pass
        elif line[0] == "@":
            spl = line.split()
            print spl
            command = spl[1]
            if command == "title":
                title = spl[2].strip("\"")+" "+spl[3].strip("\"")
            if command == "xaxis" and spl[2] == "label":
                xlabel = spl[3].strip("\"")
            if command == "yaxis" and spl[2] == "label":
                ylabel = spl[3].strip("\"")
            if command == "subtitle":
                subtitle = spl[2].strip("\"")
        else:
            nums = line.split()
            xs.append(float(nums[0]))
            ys.append(float(nums[1]))
    plt.plot(xs, ys)
    print(title, xlabel, ylabel, subtitle)
    plt.title = title
    plt.xlabel = xlabel
    plt.ylabel = ylabel
    plt.subtitle = subtitle
    plt.show()
