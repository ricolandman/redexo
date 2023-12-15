__all__ = ['OutlierFlaggingModule', 'PolynomialContinuumRemovalModule',
           'FlagAbsorptionEmissionModule',
           'SavGolContinuumRemovalModule', 'GaussianContinuumRemovalModule']
from .base import Module
import numpy as np
import matplotlib.pyplot as plt


class FlagAbsorptionEmissionModule(Module):
    '''
    Module for flagging telluric absorption or emission lines.
    '''
    def initialise(self, flux_lower_limit=0, flux_upper_limit=np.inf,
                   relative_to_continuum=True, contin_poly_order=2):
        """
        Parameters
        ----------
        flux_lower_limit: float
            All values lower will be flagged.
        flux_upper_limit: float
            All values higher will be flagged.
        relative_to_continuum: bool
            Wether the flux limits are given relative to the continuum.
        contin_poly_order: int
            Order of the polynomial to determine the continuum

        Returns
        -------
        NoneType
            None
        """
        self.flux_lower_limit = flux_lower_limit
        self.flux_upper_limit = flux_upper_limit
        self.relative_to_continuum = relative_to_continuum
        self.contin_poly_order = contin_poly_order

    def process(self, dataset, order_idx, debug=False):
        if self.relative_to_continuum:
            continuum_removed_data = PolynomialContinuumRemovalModule(
                poly_order=self.contin_poly_order)(dataset.copy(), order_idx,
                                                   debug=debug)
            spec_norm = np.nanmedian(continuum_removed_data.spec, axis=0)
        else:
            spec_norm = np.nanmedian(dataset.spec, axis=0)
        mask = (spec_norm > self.flux_lower_limit) * \
               (spec_norm < self.flux_upper_limit)
        if debug:
            print('Masking {0:.2f}% of the data'.format(
                np.sum(~mask)/mask.size*100))
        dataset.spec[:, ~mask] = np.nan
        return dataset


class OutlierFlaggingModule(Module):
    '''
    Module for flagging outliers.
    '''
    def initialise(self, sigma=5):
        """
        Parameters
        ----------
        sigma: float
            Separation from the mean in standard deviations above which
            values will be flagged.
        Returns
        -------
        NoneType
            None
        """
        self.sigma = sigma

    def process(self, dataset, order_idx, debug=False):
        std = np.nanstd(dataset.spec)
        mean = np.nanmean(dataset.spec)
        outliers = (np.abs(dataset.spec - mean)/std) > self.sigma
        if debug:
            print('Masking {0:.2f}% of the data'.format(
                np.sum(outliers) / outliers.size*100))
        dataset.spec[outliers] = np.nan
        return dataset


class PolynomialContinuumRemovalModule(Module):
    '''
    Module for removing the continuum of the spectrum using a
    polynomial fit.
    '''
    def initialise(self, poly_order=3):
        """
        Parameters
        ----------
        poly_order: int
            Order of the polynomial used to determine the continuum.
        Returns
        -------
        NoneType
            None
        """
        self.poly_order = poly_order

    def process(self, dataset, order_idx, debug=False):
        try:
            for exp in range(dataset.num_exposures):
                nans = np.isnan(dataset.spec[exp])
                cont_model = np.poly1d(np.polyfit(dataset.wavelengths[exp][~nans],
                                                dataset.spec[exp][~nans],
                                                self.poly_order))
                continuum = cont_model(dataset.wavelengths[exp][~nans])
                dataset.spec[exp][~nans] = dataset.spec[exp][~nans]/continuum
                dataset.errors[exp][~nans] = dataset.errors[exp][~nans]/continuum
            return dataset
        except:
            print('Error for order ', order_idx)
            exit()


class SavGolContinuumRemovalModule(Module):
    '''
    Module for removing the continuum of the spectrum using a
    Savitsky-Golay filter.
    '''
    def initialise(self, window_length=301, poly_order=3):
        self.window_length = window_length
        self.poly_order = poly_order

    def process(self, dataset, order_idx, debug=False):
        """
        Parameters
        ----------
        window_length: int
            Window length used in the Savitsky-Golay filter.
        poly_order: int
            Order of the polynomial used to determine the continuum.
        Returns
        -------
        NoneType
            None
        """
        from scipy.signal import savgol_filter
        for exp in range(dataset.num_exposures):
            nans = np.isnan(dataset.spec[exp])
            continuum = savgol_filter(dataset.spec[exp][~nans],
                                      self.window_length, self.poly_order)
            dataset.spec[exp][~nans] = dataset.spec[exp][~nans]/continuum
            dataset.errors[exp][~nans] = dataset.errors[exp][~nans]/continuum
        return dataset


class GaussianContinuumRemovalModule(Module):
    '''
    Module for removing the continuum of the spectrum using a
    gaussian filter.
    '''
    def initialise(self, binsize=600):
        """
        Parameters
        ----------
        binsize: int
            Full-width half maximum of the gaussian filter.
        Returns
        -------
        NoneType
            None
        """
        self.binsize = binsize

    def process(self, dataset, order_idx, debug=False):
        from scipy.signal import fftconvolve, windows
        for exp in range(dataset.num_exposures):
            spec, errors = dataset.spec[exp], dataset.errors[exp]
            nans = np.isnan(spec)

            sigma = 0.5 * self.binsize / np.sqrt(2 * np.log(2))
            sigma = int(sigma)
            n = 4
            nx = spec[~nans].shape
            gauss = windows.gaussian(int(2*n*sigma+1), sigma) / \
                (sigma*np.sqrt(2.0*np.pi))
            smoothed = fftconvolve(spec[~nans], gauss, mode="same", axes=-1)
            integral = gauss.sum()
            erf_vals = gauss[:n*sigma].sum() + gauss[n*sigma:-1].cumsum()
            scaled = np.ones(nx)
            scaled[:n*sigma] = erf_vals
            scaled[-n*sigma:] = erf_vals[::-1]
            scaled[n*sigma:-n*sigma] = integral
            hipassed = np.tile(np.nan, spec.shape)
            hipassed[~nans] = spec[~nans]*scaled/(smoothed+1.0e-250)
            hipassed_errors = np.tile(np.nan, spec.shape)
            hipassed_errors[~nans] = errors[~nans]*scaled/(smoothed+1.0e-250)
            dataset.spec[exp] = hipassed
            dataset.errors[exp] = hipassed_errors
        return dataset
