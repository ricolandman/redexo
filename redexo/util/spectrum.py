import numpy as np
import astropy.units as u


class Spectrum(np.ndarray):
    """
    Class for storing spectra. It is derived from np.ndarray to
    allow for easy use in numpy functions.
    """
    def __new__(cls, arr, wavelengths):
        spec = np.asarray(arr).view(cls)
        spec.wavelengths = wavelengths
        return spec

    def __array_finalize__(self, spec):
        if spec is None:
            return
        self.wavelengths = getattr(spec, 'wavelengths', None)

    def at(self, new_wavelengths):
        """
        Method for evaluating the spectrum at new wavelengths.
        """
        new_spec = np.interp(new_wavelengths, self.wavelengths, self)
        return Spectrum(new_spec, new_wavelengths)


def convolve_to_resolution(spec, out_res, in_res=None):
    """
    Convolve the input spectrum to a lower resolution.
    Parameters
    ----------
    spec : ``redexo.Spectrum``
        Spectrum to convolve to a specific resolution
    out_res : float
        Desired output spectral resolution
    in_res : float
        Resolution of the input spectrum. If None this will be
        determined from the wavelength sampling.
    ----------
    Returns
    ----------
    Convolved spectrum
    """
    from scipy.ndimage import gaussian_filter
    in_wlen = spec.wavelengths
    in_flux = spec
    if isinstance(in_wlen, u.Quantity):
        in_wlen = in_wlen.to(u.nm).value
    if in_res is None:
        in_res = np.mean((in_wlen[:-1]/np.diff(in_wlen)))
    # Convert FWHM to standard deviation
    sigma_LSF = np.sqrt(1./out_res**2-1./in_res**2)/(2.*np.sqrt(2.*np.log(2.)))

    spacing = np.mean(2.*np.diff(in_wlen) /
                      (in_wlen[1:]+in_wlen[:-1]))

    # Calculate the sigma to be used in the gauss filter in pixels
    sigma_LSF_gauss_filter = sigma_LSF / spacing

    result = np.tile(np.nan, in_flux.shape)
    nans = np.isnan(spec)

    result[~nans] = gaussian_filter(in_flux[~nans],
                                    sigma=sigma_LSF_gauss_filter,
                                    mode='reflect')
    return Spectrum(result, spec.wavelengths)