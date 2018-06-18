import pandas as pd

# kids_limited_by_gaia_path = '/media/snakoneczny/data/KiDS/KiDS.DR3.x.GAIA.DR2.coord.csv'
# gaia_limited_by_kids_path = '/media/snakoneczny/data/GAIA/GAIA.DR2.limited.KiDS.DR3.csv'
# out_path = '/media/snakoneczny/data/KiDS/KiDS.DR3.x.GAIA.DR2.cols.csv'

ds_1_path = '/media/snakoneczny/data/KiDS/KiDS.DR3.x.GAIA.DR2.coord.csv'
ds_2_path = '/media/snakoneczny/data/GAIA/GAIA.DR2.limited.KiDS.DR3.csv'
out_path = '/media/snakoneczny/data/KiDS/KiDS.DR3.x.GAIA.DR2.coord.csv'

# Init file with a header
# columns_1 = pd.read_csv(ds_1_path, index_col=0, nrows=1).columns
# columns_2 = pd.read_csv(ds_2_path, index_col=0, nrows=1).columns

ds_1 = pd.read_csv(ds_1_path)
ds_2 = pd.read_csv(ds_2_path)

# print(ds_1.shape)
# print(ds_2.shape)

# ds_2_cols_to_use = ds_2.columns.difference(ds_1.columns)
# ds_merged = pd.merge(ds_1, ds_2[ds_2_cols_to_use], left_index=True, right_index=True, how='outer')

# print(sorted(ds_1['source_id'].head(10)))
# print(sorted(ds_2['source_id'].head(10)))

ds_merged = pd.merge(ds_1, ds_2)

ds_merged.to_csv(out_path, index=False)
