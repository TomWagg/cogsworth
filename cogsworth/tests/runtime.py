"""This file is used to test the runtime scaling of cogsworth when using multiple cores"""

import time
import cogsworth
import argparse
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the runtime scaling of cogsworth")
    parser.add_argument("-i", "--input", type=str, default=None, help="Input file to load in")
    parser.add_argument("-n", "--nbin", type=int, nargs="+", default=[10, 1000, 100000],
                        help="Number of binaries to simulate")
    parser.add_argument("-p", "--processes", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64, 128], help="Number of processes to use")
    args = parser.parse_args()

    if args.input is None and os.path.exists("runtime_base.h5"):
        args.input = "runtime_base.h5"

    if args.input is None:
        print("Creating a base population")
        p = cogsworth.pop.Population(100)
        p.sample_initial_binaries()
        p.sample_initial_galaxy()
        p.save("runtime_base.h5")
    else:
        # load in a population of pre-sampled binaries
        print("Loading in a base population")
        p = cogsworth.pop.load(args.input, parts=["initial_binaries", "initial_galaxy"])

    runtimes = np.zeros((len(args.nbin), len(args.processes)))

    for i, nbin in enumerate(args.nbin):
        print(f"Running {nbin} binaries")
        for j, nproc in enumerate(args.processes):
            print(f"Running with {nproc} processes")
            p_subset = p[:nbin]
            p_subset.processes = nproc
            print(len(p_subset), p_subset.n_binaries_match, p_subset.n_binaries)
            start = time.time()
            p.perform_stellar_evolution()
            p.perform_galactic_evolution(progress_bar=False)
            end = time.time()

            runtimes[i, j] = end - start

    np.save("runtimes.npy", runtimes)
