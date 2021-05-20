__all__ = ["load_carmenes_data"]
import glob
from ..core import Dataset
from astropy.io import fits

def load_carmenes_data(folder, target=None, spectral_orders=None, skip_exposures = [], header_info={}):
    files = glob.glob(folder+'*nir_A.fits')
    print('Loading {0} files...'.format(len(files)))
    files = sorted(files)
    dataset = Dataset(target)

    for i,fname in enumerate(files):
        if not i in skip_exposures:
            data = fits.open(fname)

            vbar = data[0].header['HIERARCH CARACAL BERV']
            if spectral_orders is None:
                wl = data[4].data
                spectrum = data[1].data
                errors = data['SIG'].data
            else:
                wl = data[4].data[spectral_orders]
                spectrum = data[1].data[spectral_orders]
                errors = data['SIG'].data[spectral_orders]

            BJD = (data[0].header['HIERARCH CARACAL BJD'] + 2400000)

            header_data = {str(key): data[0].header[val] for key, val in header_info.items()}
            dataset.add_exposure(spectrum, wl=wl, errors=errors, vbar=vbar, obstime=BJD, **header_data)
    return dataset


def load_harps_data(folder):
    raise NotImplementedError

def load_espresso_data(folder):
    raise NotImplementedError

def load_crires_data(folder):
    raise NotImplementedError


