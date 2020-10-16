import os

from env_config import DATA_PATH
from utils import logger, save_fits
from data import merge_specialized_catalogs, add_subset_info, add_shape_info


# Input estimations
ctlg_clf_path = 'KiDS/DR4/catalogs/KiDS_DR4_x_SDSS_DR14_ann_clf_f-all__2020-06-08_17:07:35.fits'
ctlg_z_qso_path = 'KiDS/DR4/catalogs/KiDS_DR4_x_SDSS_DR14_ann_z_f-all_spec-qso__2020-06-08_16:22:38.fits'

# Output paths
out_catalog_qso_path = 'KiDS/DR4/catalogs/published/KiDS_DR4_QSO_candidates_MORE.fits'
out_catalog_path = 'KiDS/DR4/catalogs/published/KiDS_DR4_all_ML_estimates.fits'

# QSO candidates parameters
min_qso_proba = 0.90
max_r = 25

columns_to_publish = ['ID', 'RAJ2000', 'DECJ2000', 'MAG_GAAP_r', 'CLASS_STAR', 'MASK',
                      'GALAXY_PHOTO', 'QSO_PHOTO', 'STAR_PHOTO', 'CLASS_PHOTO',
                      'Z_PHOTO_QSO', 'Z_PHOTO_STDDEV_QSO', 'SUBSET']

# Read and merge classification with photo-zs
catalog = merge_specialized_catalogs(os.path.join(DATA_PATH, ctlg_clf_path), os.path.join(DATA_PATH, ctlg_z_qso_path))
logger.info('catalog original size: {}'.format(catalog.shape))

# Additional columns
catalog = add_subset_info(catalog, extra_info=False)
catalog = add_shape_info(catalog)
catalog = catalog.rename(columns={'shape': 'SHAPE', 'subset': 'SUBSET', 'Z_PHOTO': 'Z_PHOTO_QSO',
                                  'Z_PHOTO_STDDEV': 'Z_PHOTO_STDDEV_QSO'})

# QSO candidates catalog
catalog_qso = catalog.loc[
    (catalog['SHAPE'] != 'not known') &
    (catalog['MAG_GAAP_r'] < max_r) &
    (catalog['QSO_PHOTO'] > min_qso_proba)
]
logger.info('catalog QSO size: {}'.format(catalog_qso.shape))

# Save QSO candidates
save_fits(catalog_qso[columns_to_publish], os.path.join(DATA_PATH, out_catalog_qso_path))
logger.info('QSO catalog saved to: {}'.format(out_catalog_qso_path))

# Save all ML estimates
# save_fits(catalog[columns_to_publish], os.path.join(DATA_PATH, out_catalog_qso_path))
# logger.info('catalog saved to: {}'.format(out_catalog_qso_path))
