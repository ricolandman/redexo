__all__ = ["load_carmenes_data"]
import glob
from ..core import Dataset
from astropy.io import fits
from astropy import time
from astropy.coordinates import EarthLocation
import astropy.units as u
from scipy.interpolate import interp1d
import numpy as np


def load_carmenes_data(folder, target=None, spectral_orders=None,
                       skip_exposures=[], header_info={}):
    """
    Function for loading CARMENES data.
    """
    files = glob.glob(folder+'*nir_A.fits')
    print('Loading {0} files...'.format(len(files)))
    files = sorted(files)
    dataset = Dataset(target)

    for i,fname in enumerate(files):
        if i not in skip_exposures:
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

            header_data = {str(key): data[0].header[val] for
                           key, val in header_info.items()}
            dataset.add_exposure(spectrum, wl=wl, errors=errors, vbar=vbar,
                                 obstime=BJD, **header_data)
    return dataset


def load_crires_data(fpath, nod_ab='both', interpolate_to_common_wl=True,
                     target=None, spectral_orders=None, skip_exposures=[],
                     header_info={}):
    """
    Function for loading CRIRES+ data.
    Parameters
    ----------
    fpath : str
        Path to directory with reduced spectra.
    interpolate_to_common_wl : bool
         If True, this will interpolate the different nodding positions to the
         same wavelength grid
    target : ``redexo.Planet``
        System to use for calculation of barycentric velocity correction.
    spectral_orders: list
        List of spectral orders to use.
    skip_exposures: list
        List of exposures to skip.
    header_info : dict
        Dictionary with header keywords to store.
    ----------
    Returns
    ----------
    Returns
    dataset : ``redexo.Dataset``
        Dataset with the exposures in the specified directory.
    """
    dataset = Dataset()
    if nod_ab == 'both':
        files = sorted(glob.glob(fpath+"/*.fits"))
    else:
        files = sorted(glob.glob(fpath+f"/*{nod_ab}*.fits"))
    for i, fname in enumerate(files):
        if i in skip_exposures:
            continue
        spec, err, wl, header = load_1d_crires_spectra(fname)
        time_mjd = header['MJD-OBS']

        time_bjd = time.Time(time_mjd, format='mjd').jd
        paranal = EarthLocation.of_site('paranal')
        if target.coords is not None:
            sc = target.coords
            barycorr = sc.radial_velocity_correction(obstime=time.Time(
                time_mjd, format='mjd'), location=paranal).to(u.km/u.s).value
        else:
            print(("WARNING: no coordinates provided, so not doing"
                   " barycentric correction"))
            barycorr = 0
        if i == 0:
            wl0 = wl
        if interpolate_to_common_wl:
            spec = np.array([interp1d(wl[order], spec[order], axis=-1,
                                      bounds_error=False)(wl0[order])
                             for order in range(wl0.shape[0])])
            err = np.array([interp1d(wl[order], err[order], axis=-1,
                                     bounds_error=False)(wl0[order])
                            for order in range(wl0.shape[0])])

        dataset.add_exposure(spec[spectral_orders], wl=wl[spectral_orders],
                             errors=err[spectral_orders], vbar=barycorr,
                             obstime=time_bjd)
    return dataset


def load_1d_crires_spectra(filepath):
    hdu_list = fits.open(filepath)
    N_orders = 8
    N_det = 3
    all_spec = []
    all_err = []
    all_wl = []
    for det_idx in np.arange(N_det):
        for order_idx in range(N_orders):
            all_spec.append(
                hdu_list[det_idx + 1].data[f"{order_idx+2:02d}_01_SPEC"])
            all_err.append(
                hdu_list[det_idx + 1].data[f"{order_idx+2:02d}_01_ERR"])
            all_wl.append(
                hdu_list[det_idx + 1].data[f"{order_idx+2:02d}_01_WL"])
    return np.array(all_spec), np.array(all_err), np.array(all_wl),\
        hdu_list[0].header


def load_harps_data(folder):
    raise NotImplementedError


def load_espresso_data(folder):
    raise NotImplementedError
