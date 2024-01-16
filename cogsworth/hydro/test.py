import cogsworth

star_snap = cogsworth.hydro.pop.FIRESnapshot(snap_dir="../../data", snap_num=600)
fire_pot = cogsworth.hydro.potential.get_snapshot_potential(snap_dir="../../data", snap_num=600)

p = cogsworth.hydro.pop.FIREPopulation(star_snap, galactic_potential=fire_pot,
                                      subset=[0, 1, 2], max_ev_time=star_snap.snap_time)
p.create_population()

print(p._initial_binaries)
print(p.mass_binaries + p.mass_singles, star_snap.m[[0, 1, 2]].sum())
