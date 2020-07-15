import os

from env_config import DATA_PATH
from utils import logger
from data import merge_specialized_catalogs, \
    get_mag_str
from evaluation import get_v_max_weights

m_max = 25.02

ctlg_clf_path = os.path.join(
    DATA_PATH, 'KiDS/DR4/catalogs/KiDS_DR4_x_SDSS_DR14_ann_clf_f-all__2020-06-08_17:07:35.fits')
ctlg_z_qso_path = os.path.join(
    DATA_PATH, 'KiDS/DR4/catalogs/KiDS_DR4_x_SDSS_DR14_ann_z_f-all_spec-qso__2020-06-08_16:22:38.fits')

catalog_clf_path = os.path.join(
    DATA_PATH, 'KiDS/DR4/catalogs/KiDS_DR4_x_SDSS_DR14_ann_clf_f-all__2020-06-08_17:07:35.fits')
catalog_z_qso_path = os.path.join(
    DATA_PATH, 'KiDS/DR4/catalogs/KiDS_DR4_x_SDSS_DR14_ann_z_f-all_spec-qso__2020-06-08_16:22:38.fits')
weights_output_path = os.path.join(
    DATA_PATH, 'KiDS/DR4/v_weights/KiDS_DR4_x_SDSS_DR14_ann_f-all__qso_m-max-{}__2020-06-08.csv'.format(int(m_max)))

catalog = merge_specialized_catalogs(ctlg_clf_path, ctlg_z_qso_path)
logger.info('Catalog shape: {}'.format(catalog.shape))

qso_kids_photo = catalog.loc[catalog['CLASS_PHOTO'] == 'QSO']
logger.info('Catalog QSO shape: {}'.format(qso_kids_photo.shape))

# Get V/V_max weights
qso_kids_photo['v_weight'] = get_v_max_weights(qso_kids_photo[get_mag_str('r')], qso_kids_photo['Z_PHOTO'],
                                               m_max=m_max, processes=24)
logger.info('V/V_max weights calculated for {} objects'.format(qso_kids_photo.shape[0]))

# Save the weights
qso_kids_photo[['ID', 'v_weight']].to_csv(weights_output_path, index=False)
logger.info('V/V_max weights saved to: {}'.format(weights_output_path))
