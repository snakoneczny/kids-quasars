import gc

import pandas as pd
from tqdm import tqdm

subset_id_path = '/media/snakoneczny/data/KiDS/KiDS.DR3.x.GAIA.DR2.coord.csv'
data_path = '/media/snakoneczny/data/GAIA/GAIA.DR2.csv'
out_path = '/media/snakoneczny/data/GAIA/GAIA.DR2.limited.KiDS.DR3.csv'

chunk_size = 10000000

subset_id = pd.read_csv(subset_id_path, usecols=['source_id'])

# Init file with a header
columns = pd.read_csv(data_path, index_col=0, nrows=1).columns  # TODO: remove index_col when other files fixed
# pd.DataFrame(columns=columns).to_csv(out_path, index=False)

for data_chunk in tqdm(pd.read_csv(data_path, usecols=columns, chunksize=chunk_size), desc='Extracting subset'):

    print(sorted(data_chunk['source_id'].head(10)))
    print(sorted(subset_id['source_id'].head(10)))

    tmp = data_chunk.loc[data_chunk['source_id'].isin(subset_id['source_id'])]

    print(sorted(tmp['source_id'].head(10)))

    exit()

    data_chunk.loc[data_chunk['source_id'].isin(subset_id['source_id'])].to_csv(out_path, mode='a', header=False,
                                                                                index=False)

    gc.collect()
