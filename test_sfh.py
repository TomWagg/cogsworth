import cogsworth
from cogsworth.sfh import CompositeStarFormationHistory
import astropy.units as u
import matplotlib as mpl
import gala.potential as gp

# TITLE: burst composite test
# s = CompositeStarFormationHistory(
#     components=[
#         cogsworth.sfh.ConstantUniformDisc(t_burst=12 * u.Gyr, R_max=10 * u.kpc, z_max=0.5 * u.kpc, Z_all=0.02),
#         cogsworth.sfh.ConstantUniformDisc(t_burst=6 * u.Gyr, R_max=15 * u.kpc, z_max=2.0 * u.kpc, Z_all=0.01),
#     ],
#     component_ratios=[0.7, 0.3]
# )

# s.sample(10000)

# s.plot(colour_by="tau", s=10)


# s1 = cogsworth.sfh.ConstantPlummerSphere(
#     tau_min=0 * u.Gyr,
#     tau_max=12 * u.Gyr,
#     Z_all=0.02,
#     M=1e10 * u.Msun,
#     a=5 * u.kpc,
#     r_trunc=10 * u.kpc
# )

# s2 = cogsworth.sfh.BurstUniformDisc(
#     t_burst=6 * u.Gyr,
#     R_max=15 * u.kpc,
#     z_max=2.0 * u.kpc,
#     Z_all=0.01
# )

# s = 6e10 * s1 + s2 * 1e9
# print(s)

# s.sample(10000)
# s.plot(colour_by="tau", s=10)

# s1 = cogsworth.sfh.SandersBinney2015(potential=gp.MilkyWayPotential(), verbose=True)
# s2 = cogsworth.sfh.ConstantPlummerSphere(tau_min=10. * u.Gyr, tau_max=12 * u.Gyr, Z_all=1e-3, M=1e10 * u.Msun, a=2 * u.kpc, r_trunc=10 * u.kpc)

# s = 5e10 * s1 + 1e10 * s2
# print(s)

# s.sample(50000)

# s.plot(colour_by="tau", s=10)

# for component in s.components:
#     print(component)
#     component.plot(colour_by="tau", s=10)

# s.plot(colour_by="Z", s=10, cbar_norm=mpl.colors.LogNorm(vmin=1e-4, vmax=0.03))

# s.save("test.h5")

# s_loaded = CompositeStarFormationHistory.from_file("test.h5")

# print(s_loaded)

# s_loaded_mask = s_loaded[s_loaded.tau < 3 * u.Gyr]
# s_loaded_mask.plot(colour_by="tau", s=10)

p = cogsworth.pop.Population(
    n_binaries=10, processes=1, final_kstar1=[13, 14],
    use_default_BSE_settings=True
)
p.create_population()

print(p.final_bpp)