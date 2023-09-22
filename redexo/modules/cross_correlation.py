__all__ = ['CrossCorrelationModule', 'CrossCorrelationModuleFast']
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
        for exp in range(dataset.num_exposures):
            nans = np.isnan(dataset.spec[exp])
            wl_exp = dataset.wavelengths[exp][~nans]
            spec_exp = dataset.spec[exp][~nans]
            if self.empirical_error:
                errors_exp = errors_est[exp]*np.nanmean(
                    dataset.errors[exp][~nans])
            else:
                errors_exp = dataset.errors[exp][~nans]
            # Calculate grid for shifted wavelengths
            shifted_wavelengths = wl_exp*self.beta_grid[:, np.newaxis]

            shifted_template = interp1d(self.template_wl, self.template,
                                        bounds_error=True)(shifted_wavelengths)
            # Remove mean from each template
            shifted_template = shifted_template.T - np.mean(shifted_template,
                                                            axis=1)
            # Calculate CCF using matrix-vector-multiplication
            if self.error_weighted:
                CCF[exp] = (spec_exp/errors_exp**2).dot(shifted_template)
            else:
                CCF[exp] = spec_exp.dot(shifted_template)
        rv_matrix = np.tile(self.rv_grid[np.newaxis, np.newaxis, :],
                            (dataset.num_exposures, dataset.num_orders, 1))
        res = CCF_Dataset(spec=CCF, rv_grid=rv_matrix, vbar=dataset.vbar,
                          obstimes=dataset.obstimes)
        return res


class CrossCorrelationModuleFast(Module):
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
        print('Template:', shifted_template.shape)
        # Calculate CCF using matrix-vector-multiplication
        if self.error_weighted:
            CCF = (data/errors**2).dot(shifted_template)
        else:
            CCF = (data).dot(shifted_template)
        print('CCF:', CCF.shape)
        rv_matrix = np.tile(self.rv_grid[np.newaxis, np.newaxis, :],
                            (dataset.num_exposures, dataset.num_orders, 1))
        res = CCF_Dataset(spec=CCF, rv_grid=rv_matrix,
                          vbar=dataset.vbar,
                          obstimes=dataset.obstimes)
        return res
