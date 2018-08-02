import pandas as pd

ds_1_path = '/media/snakoneczny/data/KiDS/KiDS.DR3.x.SDSS.DR14.cols.csv'
ds_2_path = '/media/snakoneczny/data/SDSS/SDSS.DR14.x.GAIA.DR2.cols.csv'
out_path = '/media/snakoneczny/data/KiDS/KiDS.DR3.x.SDSS.DR14.x.GAIA.DR2.cols.csv'
id_column = 'SPECOBJID'

ds_1 = pd.read_csv(ds_1_path)
ds_2 = pd.read_csv(ds_2_path)

ds_1 = ds_1.drop('Separation', axis=1)
ds_2 = ds_2.drop('Separation', axis=1)

ds_1_x_ds_2 = pd.merge(ds_1, ds_2)

assert ds_1_x_ds_2.shape[0] != 0
print(ds_1_x_ds_2.shape)

ds_1_x_ds_2.to_csv(out_path, index=False)
