__all__ = ['CrossCorrelationModule', 'LikelihoodMapModule']
from .base import Module
from ..core.dataset import CCF_Dataset
import astropy.constants as const
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt


class CrossCorrelationModule(Module):
    '''
    Module for calculating the CCFs.
    '''
    def initialise(self, template, template_wl, rv_range, drv,
                   error_weighted=True, empirical_error=False):
        """
        Parameters
        ----------
        template: np.ndarray
            Model template to use for cross-correlation.
        template_wl: np.ndarray
            Wavelengths at which the template is sampled.
        rv_range: np.ndarray
            Radial velocity range at which the CCF is evaluated.
        drv: float
            Radial velocity step size to use for the CCF calculation.
        error_weighted: bool
            If ``True``, module will do a weighted CCF.
        empirical_err: bool
            If ``True``, module will empirically estimate the error of
            each wavelength bin by taking the standard deviation over time.
        Returns
        -------
        NoneType
            None
        """
        self.template = template
        self.template_wl = template_wl
        self.rv_range = rv_range
        self.drv = drv
        self.empirical_error = empirical_error
        self.rv_grid = np.arange(-rv_range, rv_range+1e-4, drv)
        self.beta_grid = 1.0-self.rv_grid/const.c.to('km/s').value
        self.error_weighted = error_weighted

    def process(self, dataset, order_idx, debug=False):
        CCF = np.zeros((dataset.num_exposures, dataset.num_orders,
                        self.rv_grid.size))
        if self.empirical_error:
            errors_est = np.nanstd(dataset.spec, axis=-1)
        nans = np.isnan(dataset.spec)
        if self.empirical_error:
            errors = errors_est[np.newaxis, :]*np.nanmean(
                dataset.errors[~nans], axis=(-1))[:, :, None]
        else:
            errors = dataset.errors
        # Calculate grid for shifted wavelengths
        errors[nans] = np.inf
        data = dataset.spec.copy()
        data -= np.nanmean(data, axis=-1)[:,:,np.newaxis]
        data[nans] = 0
        wl = dataset.wavelengths[0, :, :]
        shifted_wavelengths = wl.T * \
            self.beta_grid[None, :]

        shifted_template = interp1d(self.template_wl, self.template,
                                    bounds_error=True)(shifted_wavelengths)

        # Remove mean from each template
        shifted_template = shifted_template - np.mean(shifted_template, axis=0)
        # Calculate CCF using matrix-vector-multiplication
        if self.error_weighted:
            CCF = (data/errors**2).dot(shifted_template)
        else:
            CCF = (data).dot(shifted_template)
        rv_matrix = np.tile(self.rv_grid[np.newaxis, np.newaxis, :],
                            (dataset.num_exposures, dataset.num_orders, 1))
        res = CCF_Dataset(spec=CCF, rv_grid=rv_matrix,
                          vbar=dataset.vbar,
                          obstimes=dataset.obstimes)
        return res


class LikelihoodMapModule(Module):
    '''
    Module for calculating the likelihood map.
    '''
    def initialise(self, template, template_wl, rv_range, drv,
                   error_weighted=True, empirical_error=False):
        """
        Parameters
        ----------
        template: np.ndarray
            Model template to use for cross-correlation.
        template_wl: np.ndarray
            Wavelengths at which the template is sampled.
        rv_range: np.ndarray
            Radial velocity range at which the lnL map is evaluated.
        drv: float
            Radial velocity step size to use for the lnL map calculation.
        error_weighted: bool
            If ``True``, module will do a weighted lnL as in Gibson
            et al. 2020, while otherwise the Brogi&Line 2019 likelihood
            is used.
        empirical_err: bool
            If ``True``, module will empirically estimate the error of
            each wavelength bin by taking the standard deviation over time.
        Returns
        -------
        NoneType
            None
        """
        self.template = template
        self.template_wl = template_wl
        self.rv_range = rv_range
        self.drv = drv
        self.empirical_error = empirical_error
        self.rv_grid = np.arange(-rv_range, rv_range+1e-4, drv)
        self.beta_grid = 1.0-self.rv_grid/const.c.to('km/s').value
        self.error_weighted = error_weighted

    def process(self, dataset, order_idx, debug=False):
        lnL = np.zeros((dataset.num_exposures, dataset.num_orders,
                        self.rv_grid.size))
        if self.empirical_error:
            errors = np.nanstd(dataset.spec, axis=-1)[:, None] * \
                np.nanstd(dataset.spec, axis=0)[None, :]
        else:
            errors = dataset.errors
        # Calculate grid for shifted wavelengths
        data = dataset.spec.copy()[:, 0]
        nans = np.isnan(data)
        errors = errors[:, 0]
        errors[nans] = np.inf
        data -= np.nanmean(data, axis=-1)[:, None]
        data[nans] = 0
        wl = dataset.wavelengths[0, :, :]
        shifted_wavelengths = wl.T * \
            self.beta_grid[None, :]

        shifted_template = interp1d(self.template_wl, self.template,
                                    bounds_error=True)(shifted_wavelengths)

        # Remove mean from each template
        shifted_template = shifted_template - np.mean(shifted_template, axis=0)
        if self.error_weighted:
            chi2 = np.nansum(
                (data[:, :, None] - shifted_template[None, :, :])**2 /
                errors[:, :, None]**2, axis=1)
        else:
            chi2 = np.nansum(
                (data[:, :, None] - shifted_template[None, :, :])**2)

        N = np.sum(~np.isnan(dataset.spec[:, 0]), axis=-1)[:, None]
        lnL = -N / 2 * np.log(chi2 / N)
        lnL -= np.median(lnL, axis=1)[:, None]
        rv_matrix = np.tile(self.rv_grid[np.newaxis, np.newaxis, :],
                            (dataset.num_exposures, dataset.num_orders, 1))
        res = CCF_Dataset(spec=lnL[:, np.newaxis], rv_grid=rv_matrix,
                          vbar=dataset.vbar,
                          obstimes=dataset.obstimes)
        return res
