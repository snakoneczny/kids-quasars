import os

from utils import save_fits
from data import DATA_PATH, read_fits_to_pandas

ctlg_clf_path = os.path.join(
    DATA_PATH, 'KiDS/DR4/catalogs/KiDS_DR4_x_SDSS_DR14_ann_clf_f-all__2020-06-08_17:07:35.fits')
ctlg_z_qso_path = os.path.join(
    DATA_PATH, 'KiDS/DR4/catalogs/KiDS_DR4_x_SDSS_DR14_ann_z_f-all_spec-qso__2020-06-08_16:22:38.fits')

ctlg_clf = read_fits_to_pandas(ctlg_clf_path)
ctlg_z_qso = read_fits_to_pandas(ctlg_z_qso_path)

print(ctlg_clf.shape)

ctlg_clf[['QSO_Z_PHOTO', 'QSO_Z_PHOTO_STDDEV']] = ctlg_z_qso[['Z_PHOTO', 'Z_PHOTO_STDDEV']]

ctlg_clf = ctlg_clf[['ID', 'RAJ2000', 'DECJ2000', 'GALAXY_PHOTO', 'QSO_PHOTO', 'STAR_PHOTO',
                     'QSO_Z_PHOTO', 'QSO_Z_PHOTO_STDDEV']]

# Save FITS
catalog_path = os.path.join(DATA_PATH, 'KiDS/DR4/catalogs/KiDS_DR4_x_SDSS_DR14_ann__2020-06-08.fits')
save_fits(ctlg_clf, catalog_path)
