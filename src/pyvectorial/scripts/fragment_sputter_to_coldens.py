#!/usr/bin/env python3

import os
import sys
import pathlib
import warnings

import logging as log
import numpy as np
from argparse import ArgumentParser
import astropy.units as u
from astropy.table import QTable

# from astropy.visualization import quantity_support
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sbpy.activity as sba
import pyvectorial as pyv


def process_args():

    # Parse command-line arguments
    parser = ArgumentParser(
        usage='%(prog)s [options] [inputfile]',
        description=__doc__,
        prog=os.path.basename(sys.argv[0])
    )
    parser.add_argument('--verbose', '-v', action='count', default=0,
                        help='increase verbosity level')
    parser.add_argument(
            'fitsinput', nargs=1, help='fits file that contains calculation table'
            )  # the nargs=? specifies 0 or 1 arguments: it is optional

    args = parser.parse_args()

    # handle verbosity
    if args.verbose >= 2:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    elif args.verbose == 1:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    return args


def main():

    # # sometimes the aperture counts get a little complainy
    # warnings.filterwarnings("ignore")
    args = process_args()
    table_file = pathlib.PurePath(args.fitsinput[0])

    # read in table from FITS
    test_table = QTable.read(table_file, format='fits')

    # add some analysis data to table and sort by base_q
    print("Adding columns for model input parameters ...")
    add_vmc_columns(test_table)

    print("Computing backflow estimates ...")
    add_backflow_estimate(test_table)

    test_table.sort(keys='base_q')

    print("Constructing results table ...")
    output_table = test_table

    output_table.remove_columns(names=['b64_encoded_vmc', 'b64_encoded_coma', 'vmc_hash'])

    # qOH_vs_qH2O_plots_combined(table_file, output_table)
    backflow_plots_combined(table_file, output_table)

    print("Writing results ...")
    output_table_file = pathlib.PurePath('analysis.ecsv')
    output_table.write(output_table_file, format='ascii.ecsv', overwrite=True)

    print("Complete!")


if __name__ == '__main__':
    sys.exit(main())
