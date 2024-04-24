import os

# get the dustmap file if necessary
# import dustmaps.bayestar
# from dustmaps.std_paths import data_dir
# bayestar_path = os.path.join(data_dir(), 'bayestar', '{}.h5'.format("bayestar2019"))
# if not os.path.exists(bayestar_path):
#     dustmaps.bayestar.fetch()

# set up directory for gaiaunlimited
home_dir = os.path.expanduser('~')
gaia_unlimited_path = os.path.join(home_dir, ".gaiaunlimited")
if not os.path.isdir(gaia_unlimited_path):
    os.mkdir(gaia_unlimited_path)
