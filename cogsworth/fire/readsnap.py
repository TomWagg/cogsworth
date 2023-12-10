import os
import h5py as h5
import numpy as np
from collections import defaultdict


def read_snapshot(snap_dir, snap_num, ptype, h0=False, cosmological=True, header_only=False):
    """Read in a snapshot from a FIRE simulation

    This code is heavily based on a function written by Matt Orr, thanks Matt!

    Parameters
    ----------
    ptype : `int`
        Particle type to use
    h0 : `bool`, optional
        _description_, by default False
    cosmological : `bool`, optional
        Whether the simulation was run in cosmological mode, by default True
    header_only : `bool`, optional
        Whether to just return the header information, by default False

    Returns
    -------
    params : `dict`
        The various parameters associated with the snapshot and `self.particle_type` across every file
        from this snapshot (only returned when `header_only` is False)
    header_params : `dict`
        The header parameters for this snapshot

    Raises
    ------
    FileNotFoundError
        No snapshot file could be found
    ValueError
        No particles of type `self.particle_type` in snapshot
    """
    # by default assume it's in one file
    file_path = os.path.join(snap_dir, f"snapshot_{snap_num}.hdf5")

    # if I can't find that then let's assume it's in multiple files
    if not os.path.exists(file_path):
        file_path = os.path.join(snap_dir, f"snapshot_{snap_num}.0.hdf5")

    # if I can't find *that* then we've got a real problem and need to quit
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find snapshot {snap_num} in {snap_dir}")

    # open file and parse its header information
    file = h5.File(file_path, 'r')
    header = file["Header"].attrs

    npart = header["NumPart_ThisFile"]
    time = header["Time"]
    boxsize = header["BoxSize"]
    fixed_masses = header["MassTable"][ptype] > 0
    omega_matter = header["Omega0"] if "Omega0" in list(header.keys()) else header["Omega_Matter"]

    hinv = 1 / header["HubbleParam"] if h0 else 1
    ascale = 1.
    if cosmological:
        ascale = time
        hinv = 1 / header["HubbleParam"]
    else:
        time *= hinv
    boxsize *= hinv * ascale

    if header["NumPart_Total"][ptype] <= 0:
        file.close()
        raise ValueError(f"No particles of type {ptype} in snapshot!")

    header_params = {'k': 0, 'time': time, 'boxsize': boxsize, 'hubble': header["HubbleParam"],
                     'npart': npart, 'npartTotal': header["NumPart_Total"], 'Omega0': omega_matter}
    if header_only:
        file.close()
        return header_params

    # initialize variables to be read
    pos = np.zeros([header["NumPart_Total"][ptype], 3], dtype=np.float64)
    vel = np.copy(pos)
    ids = np.zeros([header["NumPart_Total"][ptype]], dtype=int)
    mass = np.zeros([header["NumPart_Total"][ptype]], dtype=np.float64)
    one_D_vars = defaultdict(lambda: np.zeros([header["NumPart_Total"][ptype]], dtype=np.float64))
    if (ptype == 0 or ptype == 4) and (header["Flag_Metals"] > 0):
        metal = np.zeros([header["NumPart_Total"][ptype], header["Flag_Metals"]], dtype=np.float64)

    # loop over the snapshot parts to get the different data pieces
    nL = 0
    for i_file in range(header["NumFilesPerSnapshot"]):
        if header["NumFilesPerSnapshot"] > 1:
            file.close()
            file = h5.File(os.path.join(snap_dir, f"snap_{snap_num}.{i_file}.hdf5"), 'r')

        input_struct = file
        npart = file["Header"].attrs["NumPart_ThisFile"]
        bname = f"PartType{ptype}/"

        # now do the actual reading
        if npart[ptype] > 0:
            nR = nL + npart[ptype]
            pos[nL:nR, :] = input_struct[bname+"Coordinates"]
            vel[nL:nR, :] = input_struct[bname+"Velocities"]
            ids[nL:nR] = input_struct[bname+"ParticleIDs"]
            mass[nL:nR] = input_struct[bname + "Masses"] if ~fixed_masses else header["MassTable"][ptype]

            if ptype == 0:
                one_D_vars["u"][nL:nR] = input_struct[bname+"InternalEnergy"]
                one_D_vars["rho"][nL:nR] = input_struct[bname+"Density"]
                one_D_vars["h"][nL:nR] = input_struct[bname+"SmoothingLength"]
                if "MolecularHydrogenFraction" in list(input_struct[bname].keys()):
                    one_D_vars["fH2"][nL:nR] = input_struct[bname+"MolecularHydrogenFraction"]
                if "ArtificialViscosity" in list(input_struct[bname].keys()):
                    one_D_vars["alpha"][nL:nR] = input_struct[bname+"ArtificialViscosity"]
                if "Vorticity" in list(input_struct[bname].keys()):
                    one_D_vars["vorticity"][nL:nR, :] = input_struct[bname+"Vorticity"]
                if (header["NumPart_Total"][ptype] > 0):
                    one_D_vars["ne"][nL:nR] = input_struct[bname+"ElectronAbundance"]
                    one_D_vars["nh"][nL:nR] = input_struct[bname+"NeutralHydrogenAbundance"]
                if (header["Flag_Sfr"] > 0):
                    one_D_vars["sfr"][nL:nR] = input_struct[bname+"StarFormationRate"]

            if (ptype == 0 or ptype == 4) and (header["Flag_Metals"] > 0):
                metal_t = input_struct[bname + "Metallicity"]
                if (header["Flag_Metals"] > 1):
                    if (metal_t.shape[0] != npart[ptype]):
                        metal_t = np.transpose(metal_t)
                else:
                    metal_t = np.reshape(np.array(metal_t), (np.array(metal_t).size, 1))
                metal[nL:nR, :] = metal_t

            if ptype == 4 and header["Flag_Sfr"] > 0 and header["Flag_StellarAge"] > 0:
                one_D_vars["age"][nL:nR] = input_struct[bname + "StellarFormationTime"]

            # move cursor for next iteration
            nL = nR

    # correct to same ID as original gas particle for new stars, if bit-flip applied
    if (np.min(ids) < 0) | (np.max(ids) > 1.e9):
        bad = (ids < 0) | (ids > 1.e9)
        ids[bad] += (1 << 31)

    # do the cosmological conversions on final vectors as needed
    pos *= hinv * ascale        # snapshot units are comoving
    mass *= hinv
    vel *= np.sqrt(ascale)      # remember gadget's weird velocity units!
    if ptype == 0:
        one_D_vars["rho"] *= (hinv / (ascale*hinv)**3)
        one_D_vars["h"] *= hinv * ascale
    if ptype == 4 and header["Flag_Sfr"] > 0 and header["Flag_StellarAge"] > 0 and not cosmological:
        one_D_vars["age"] *= hinv

    params = {'k': 1, 'p': pos, 'v': vel, 'm': mass, 'id': ids}
    params.update(one_D_vars)
    if ptype in [0, 4] and header["Flag_Metals"] > 0:
        params.update({'Z': metal})
    file.close()
    return params, header_params
