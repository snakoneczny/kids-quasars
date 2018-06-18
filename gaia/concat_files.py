from os import listdir
from os.path import isfile, join

import pandas as pd
from tqdm import tqdm

data_folder = '/media/snakoneczny/data/GAIA/dr2'
out_file = '/media/snakoneczny/data/GAIA/dr2_coord.csv'

# columns = ['source_id', 'ra', 'ra_error', 'dec', 'dec_error', 'parallax', 'parallax_error', 'pmra', 'pmra_error',
#            'pmdec', 'pmdec_error', 'radial_velocity', 'radial_velocity_error']

columns = ['source_id', 'ra', 'dec']

files = [join(data_folder, f) for f in listdir(data_folder) if isfile(join(data_folder, f))]

# Init file with a header
pd.DataFrame(columns=columns).to_csv(out_file, index=False)

for f in tqdm(files):
    data = pd.read_csv(f)[columns]
    data.to_csv(out_file, mode='a', header=False, index=False)
