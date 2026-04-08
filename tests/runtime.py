"""This file is used to test the runtime scaling of cogsworth when using multiple cores"""

import time
import cogsworth
import argparse
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the runtime scaling of cogsworth")
    parser.add_argument("-i", "--input", type=str, default=None, help="Input file to load in")
    parser.add_argument("-n", "--nbin", type=int, default=10000,
                        help="Number of binaries to simulate")
    parser.add_argument("-p", "--processes", type=int, default=1,
                        help="Number of processes to use")
    parser.add_argument("-w", "--weak-scaling", action="store_true",
                        help="Use weak scaling", default=False)
    args = parser.parse_args()

    if args.input is None and os.path.exists(f"runtime_base_{args.nbin}.h5"):
        args.input = f"runtime_base_{args.nbin}.h5"

    if args.input is None:
        print("Creating a base population")
        p = cogsworth.pop.Population(args.nbin, use_default_BSE_settings=True)
        p.sample_initial_binaries()
        p.sample_initial_galaxy()
        p = p[:args.nbin]
        p.save(f"runtime_base_{args.nbin}.h5")
    else:
        # load in a population of pre-sampled binaries
        print("Loading in a base population")
        p = cogsworth.pop.load(args.input, parts=["initial_binaries", "initial_galaxy"])

    # if we're testing weak scaling, we need to copy the population for each process
    if args.weak_scaling:
        copies = [p[:] for _ in range(args.processes)]
        p = cogsworth.pop.concat(*copies)

    p.processes = args.processes
    start = time.time()
    p.perform_stellar_evolution()
    cosmic_done = time.time()
    p.perform_galactic_evolution(progress_bar=False)
    gala_done = time.time()

    cosmic_time = cosmic_done - start
    gala_time = gala_done - cosmic_done
    total_time = gala_done - start

    print(f"{len(p)} binaries, with {args.processes}, took {total_time:1.2f} seconds to run ({cosmic_time:1.2f} for cosmic, {gala_time:1.2f} for gala)")

    suffix = "_weak" if args.weak_scaling else ""

    p.save(f"runtime_results/runtime_{args.nbin}_{args.processes}_pop{suffix}.h5", overwrite=True)

    np.save(f"runtime_results/runtimes_{args.nbin}_{args.processes}{suffix}.npy", [cosmic_time, gala_time, total_time])
