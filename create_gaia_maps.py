import gc
import math

import numpy as np
import pandas as pd
from tqdm import tqdm
import healpy as hp

data_path = '/media/snakoneczny/data/GAIA/GAIA.DR2.GAL.csv'
out_path = '/home/snakoneczny/workspace/kids_quasars/maps/GAIA_DR2_nside-{}'
chunk_size = 10000000
nsides = [1024, 2048]


# Set the number of sources and the coordinates for the input
npix_dict = {nside: hp.nside2npix(nside) for nside in nsides}  # 12 * nside ^ 2

# Initate the map and fill it with the values
hpxmap_dict = {nside: np.zeros(npix_dict[nside], dtype=np.float) for nside in nsides}

# Create corresponding coordinates for pixel map
lon_dict, lat_dict = {}, {}
for nside in nsides:
    lon_dict[nside], lat_dict[nside] = hp.pixelfunc.pix2ang(nside, range(npix_dict[nside]), nest=False, lonlat=True)

for data_chunk in tqdm(pd.read_csv(data_path, chunksize=chunk_size), desc='Creating maps'):

    # TODO: clean data?

    # Get galactic coordinates
    l = data_chunk['GAL_LONG']
    b = data_chunk['GAL_LAT']

    # Coordinates and the density field f
    phis = l / 180. * math.pi
    thetas = (-1. * b + 90.) / 180. * math.pi

    for nside in nsides:

        # Go from HEALPix coordinates to indices
        indices = hp.ang2pix(nside, thetas, phis, nest=False)
        for i in indices:
            hpxmap_dict[nside][i] += 1

    gc.collect()

for nside in nsides:
    np.savetxt(out_path.format(nside) + '_hpxmap.txt', hpxmap_dict[nside])
    np.savetxt(out_path.format(nside) + '_lon.txt', lon_dict[nside])
    np.savetxt(out_path.format(nside) + '_lat.txt', lat_dict[nside])
