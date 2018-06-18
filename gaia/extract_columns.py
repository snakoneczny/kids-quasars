import pandas as pd
from tqdm import tqdm


data_path = '/media/snakoneczny/data/GAIA/dr2.csv'
out_path = '/media/snakoneczny/data/GAIA/dr2_coord.csv'
columns = ['source_id', 'ra', 'dec']
chunk_size = 50000000

# Init file with a header
pd.DataFrame(columns=columns).to_csv(out_path, index=False)

for data_chunk in tqdm(pd.read_csv(data_path, chunksize=chunk_size, usecols=columns), desc='Extracting columns'):
    data_chunk.to_csv(out_path, mode='a', header=False, index=False)
