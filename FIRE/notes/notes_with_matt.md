# FIRE Notes

The simulation files are all found here (for m11h anyway)
```
/mnt/home/chayward/firesims/fire2/public_release/core/m11h_res7100
```

Matt has a bunch of code for reading these files all found under
```
/mnt/home/morr/pfh_python
```

In particular `readsnap` is a highlight, others include:

- FIREMapper (/mnt/home/morr/code/FIREMapper)

- FIRECenterer will tell you where the galaxy is and the orientation etc.

- OmegaMapper -> for finding the potential


The particle types are:
- 0 = gas
- 1 = high res DM
- 2 = low res DM
- 4 = stars

I could investigate more about how the supernovae are implemented in
    
    galaxy_sf, stellar_evolution.c

http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html#snaps-snaps