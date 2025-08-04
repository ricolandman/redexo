__all__ = ['SysRemModule', 'PCASubtractModule', 'SysRemExcludeModule',
           'SysRemOrthModule', 'JointFittingModule']
from .base import Module
from ..core.dataset import CCF_Dataset
import numpy as np
import astropy.constants as const
from scipy.interpolate import interp1d
from tqdm import tqdm


class PCASubtractModule(Module):
    '''
    Module for subtracting systematics from the data using
    a Principal Component Analysis.
    '''
    def initialise(self, number_of_modes):
        """
        Parameters
        ----------
        number_of_modes: int
            Number of PCA modes to subtract.
        Returns
        -------
        NoneType
            None
        """
        self.number_of_modes = number_of_modes
        self.per_order = True

    def process(self, dataset, order_idx, debug=False):
        spec = dataset.spec[:, 0, :]
        norm_spec = (spec - np.nanmean(spec, axis=0))
        nans = np.any(np.isnan(norm_spec), axis=0)
        norm_spec[:, nans] = np.nan
        u, s, v = np.linalg.svd(norm_spec[:, ~nans], full_matrices=False)
        s_new = np.copy(s)
        s_new[:self.number_of_modes] = 0

        spec_corr = np.dot(u, np.dot(np.diag(s_new), v))
        norm_spec[:, ~nans] = spec_corr
        dataset.spec = norm_spec[:, np.newaxis]
        return dataset
    
def get_sysrem_mode(spec, errors, max_iter=1000, a_prev = None):
    correction = np.zeros_like(spec)
    if a_prev is not None:
        a = a_prev
    else:
        a = np.ones(spec.shape[0])
    for j in range(max_iter):
        prev_correction = correction
        c = np.nansum(spec.T * a / errors.T**2, axis=1) / \
            np.nansum(a**2/errors.T**2, axis=1)
        a = np.nansum(c*spec/errors**2, axis=1) / \
            np.nansum(c**2/errors**2, axis=1)
        correction = np.dot(a[:, np.newaxis], c[np.newaxis, :])
        fractional_dcorr = np.sum(
            np.abs(correction - prev_correction)) / \
            (np.sum(np.abs(prev_correction))+1e-5)
        if j > 1 and fractional_dcorr < 1e-3:
            break
    return correction, c, a


class SysRemModule(Module):
    """
    Module for removing systematics from the data using
    SysRem
    """
    def initialise(self, number_of_modes, max_iterations_per_mode=1000,
                   mode='subtract'):
        """
        Parameters
        ----------
        number_of_modes: int
            Number of SysRem modes to remove.
        max_iterations_per_mode: int
            Maximum number of iterations before assuming convergence.
        mode: str
            If `subtract` then module will subtract the SysRem components,
            if `divide` then module will divide by the SysRem components.
        Returns
        -------
        NoneType
            None
        """
        self.number_of_modes = number_of_modes
        self.max_iterations_per_mode = max_iterations_per_mode
        self.per_order = True
        self.mode = mode

    def process(self, dataset, order_idx, debug=False):
        spec = dataset.spec[:, 0, :]
        errors = dataset.errors[:, 0, :]
        spec = (spec.T - np.nanmean(spec, axis=1)).T
        nans = np.any(np.isnan(spec), axis=0)
        spec = spec[:, ~nans]
        errors = errors[:, ~nans]
        if isinstance(self.number_of_modes, list) or \
           isinstance(self.number_of_modes, np.ndarray):
            num_modes = self.number_of_modes[order_idx]
        else:
            num_modes = self.number_of_modes
        a = np.ones(spec.shape[0])
        for i in range(num_modes):
            correction, mode, a = get_sysrem_mode(spec, errors, self.max_iterations_per_mode, a_prev=a)
            if self.mode == 'subtract':
                spec -= correction
            elif self.mode == 'divide':
                spec /= correction
                errors /= correction
        full_spec = np.tile(np.nan, dataset.spec[:, 0, :].shape)
        full_spec[:, ~nans] = spec
        dataset.spec = full_spec[:, np.newaxis]
        return dataset


class SysRemExcludeModule(Module):
    """
    Module for removing systematics from the data using
    SysRem
    """
    def initialise(self, number_of_modes,
                   target, min_delta_rv = 2,
                   max_iterations_per_mode=1000, mode='subtract'):
        """
        Parameters
        ----------
        number_of_modes: int
            Number of SysRem modes to remove.
        max_iterations_per_mode: int
            Maximum number of iterations before assuming convergence.
        mode: str
            If `subtract` then module will subtract the SysRem components,
            if `divide` then module will divide by the SysRem components.
        Returns
        -------
        NoneType
            None
        """
        self.number_of_modes = number_of_modes
        self.max_iterations_per_mode = max_iterations_per_mode
        self.per_order = True
        self.mode = mode
        self.min_delta_rv = min_delta_rv
        self.target = target

    def process(self, dataset, order_idx, debug=False):
        spec = dataset.spec[:, 0, :]
        errors = dataset.errors[:, 0, :]
        spec = (spec.T - np.nanmean(spec, axis=1)).T
        nans = np.any(np.isnan(spec), axis=0)
        spec = spec[:, ~nans]
        errors = errors[:, ~nans]
        if isinstance(self.number_of_modes, list) or \
           isinstance(self.number_of_modes, np.ndarray):
            num_modes = self.number_of_modes[order_idx]
        else:
            num_modes = self.number_of_modes

        rvs = self.target.radial_velocity(obs_time=dataset.obstimes)
        print(rvs)
        #uncorrected_spec = spec.copy()
        for i in range(num_modes):
            residual_spec = spec.copy()
            for exposure in range(spec.shape[0]):
                exp_mask = (np.abs(rvs - rvs[exposure])>self.min_delta_rv)
                a = np.ones(np.sum(exp_mask))
                reference_spec = residual_spec[exp_mask].copy()
                reference_errors = errors[exp_mask].copy()
                reference_correction, mode, a = get_sysrem_mode(
                    reference_spec, reference_errors, self.max_iterations_per_mode,
                    a_prev = a)
                correction = np.nansum(spec[exposure]*mode/errors[exposure]**2) / \
                    np.nansum(mode**2 /errors[exposure]**2) * mode
                #reference_spec -= reference_correction
                spec[exposure] -= correction
        full_spec = np.tile(np.nan, dataset.spec[:, 0, :].shape)
        full_spec[:, ~nans] = spec
        dataset.spec = full_spec[:, np.newaxis]
        return dataset


class SysRemOrthModule(Module):
    """
    Module for removing systematics from the data using
    SysRem
    """
    def initialise(self, number_of_modes,
                   target, template, template_wl,
                   min_delta_rv = 2,
                   max_iterations_per_mode=1000, mode='subtract'):
        """
        Parameters
        ----------
        number_of_modes: int
            Number of SysRem modes to remove.
        max_iterations_per_mode: int
            Maximum number of iterations before assuming convergence.
        mode: str
            If `subtract` then module will subtract the SysRem components,
            if `divide` then module will divide by the SysRem components.
        Returns
        -------
        NoneType
            None
        """
        self.number_of_modes = number_of_modes
        self.max_iterations_per_mode = max_iterations_per_mode
        self.per_order = True
        self.mode = mode
        self.min_delta_rv = min_delta_rv
        self.target = target
        self.template = template
        self.template_wl = template_wl

    def process(self, dataset, order_idx, debug=False):
        spec = dataset.spec[:, 0, :]
        errors = dataset.errors[:, 0, :]
        wl = dataset.wavelengths[:, 0, :]
        spec = (spec.T - np.nanmean(spec, axis=1)).T
        nans = np.any(np.isnan(spec), axis=0)
        spec = spec[:, ~nans]
        errors = errors[:, ~nans]
        wl = wl[:, ~nans]
        if isinstance(self.number_of_modes, list) or \
           isinstance(self.number_of_modes, np.ndarray):
            num_modes = self.number_of_modes[order_idx]
        else:
            num_modes = self.number_of_modes

        rvs = self.target.radial_velocity(obs_time=dataset.obstimes)
        a = np.ones_like(spec.shape[0])

        for i in range(num_modes):
            reference_correction, mode, a = get_sysrem_mode(
                spec, errors, self.max_iterations_per_mode)
            for exposure in range(spec.shape[0]):
                if self.target.in_transit(obs_time=dataset.obstimes[exposure]):
                    beta = 1 - rvs[exposure]/const.c.to('km/s').value
                    wl_new = beta * wl[exposure]
                    exp_planet_signal = interp1d(
                        self.template_wl, self.template, bounds_error=True,
                        fill_value=np.median(self.template))(wl_new)
                    orth_mode = mode - np.sum(mode * exp_planet_signal/errors[exposure]**2)/ \
                        np.sum(exp_planet_signal**2/errors[exposure]**2) * mode
                else:
                    orth_mode = mode
                correction = np.nansum(spec[exposure]*orth_mode/errors[exposure]**2) / \
                    np.nansum(mode**2 /errors[exposure]**2) * orth_mode
                #reference_spec -= reference_correction
                spec[exposure] -= correction
        full_spec = np.tile(np.nan, dataset.spec[:, 0, :].shape)
        full_spec[:, ~nans] = spec
        dataset.spec = full_spec[:, np.newaxis]
        return dataset


class JointFittingModule(Module):
    """
    Module for removing systematics from the data using
    SysRem
    """
    def initialise(self, number_of_modes,
                   target, template, template_wl,
                   rv_range, drv=1, max_iterations_per_mode=1000,
                   error_weighted=False,
                   mode='subtract'):
        """
        Parameters
        ----------
        number_of_modes: int
            Number of SysRem modes to remove.
        max_iterations_per_mode: int
            Maximum number of iterations before assuming convergence.
        mode: str
            If `subtract` then module will subtract the SysRem components,
            if `divide` then module will divide by the SysRem components.
        Returns
        -------
        NoneType
            None
        """
        self.number_of_modes = number_of_modes
        self.max_iterations_per_mode = max_iterations_per_mode
        self.per_order = True
        self.mode = mode
        self.target = target
        self.template = template
        self.template_wl = template_wl
        self.rv_grid = np.arange(-rv_range, rv_range+1e-4, drv)
        self.error_weighted = error_weighted

    def process(self, dataset, order_idx, debug=False):
        spec = dataset.spec[:, 0, :]
        errors = dataset.errors[:, 0, :]
        wl = dataset.wavelengths[:, 0, :]
        spec = (spec.T - np.nanmean(spec, axis=1)).T
        nans = np.any(np.isnan(spec), axis=0)
        spec = spec[:, ~nans]
        errors = errors[:, ~nans]
        wl = wl[:, ~nans]
        if isinstance(self.number_of_modes, list) or \
           isinstance(self.number_of_modes, np.ndarray):
            num_modes = self.number_of_modes[order_idx]
        else:
            num_modes = self.number_of_modes

        rvs = self.target.radial_velocity(obs_time=dataset.obstimes)
        a = np.ones_like(spec.shape[0])
        residual_spec = spec.copy()
        modes = np.zeros((num_modes, spec.shape[1]))

        for i in range(num_modes):
            correction, mode, a = get_sysrem_mode(
                residual_spec, errors, self.max_iterations_per_mode)
            residual_spec -= correction
            modes[i] = mode
        
        ccf = np.zeros((spec.shape[0], self.rv_grid.size))
        for exposure in tqdm(range(dataset.num_exposures)):
            for rv_i, rv in enumerate(self.rv_grid):
                shifted_wavelengths = wl[exposure] * (1. - rv/const.c.to('km/s').value)
                shifted_template = interp1d(self.template_wl, self.template,
                                    bounds_error=True)(shifted_wavelengths)
                shifted_template = shifted_template - np.mean(shifted_template, axis=0)
                #if self.target.in_transit(obs_time=dataset.obstimes[exposure]):
                if True:
                    M = np.concatenate([shifted_template[None, :], modes], axis=0)
                    # Normalize M
                    M = M / np.nanstd(M, axis=1)[:, None]
                else:
                    M = modes
                M = np.nan_to_num(M)
                if self.error_weighted:
                    lhs = np.dot(M, M.T/errors[exposure][:, None]**2)
                    rhs = np.dot(M, spec[exposure]/errors[exposure]**2)
                else:
                    lhs = np.dot(M, M.T)
                    rhs = np.dot(M, spec[exposure])
                linear_params = np.linalg.lstsq(lhs, rhs, rcond=1e-10)[0]
                ccf[exposure, rv_i] = linear_params[0]

        ccf = ccf - np.mean(ccf, axis=1)[:, None]
        rv_matrix = np.tile(self.rv_grid[np.newaxis, np.newaxis, :],
                            (dataset.num_exposures, dataset.num_orders, 1))
        res = CCF_Dataset(spec=ccf, rv_grid=rv_matrix,
                          vbar=dataset.vbar,
                          obstimes=dataset.obstimes)
        return res
