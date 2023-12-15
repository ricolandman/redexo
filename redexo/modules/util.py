__all__ = ['ShiftRestFrameModule', 'InjectSignalModule', 'CoAddExposures',
           'CoAddOrders', 'make_kp_vsys_map', 'highpass_gaussian',
           'broaden']
from .base import Module
from ..core.dataset import Dataset, CCF_Dataset
import numpy as np
import astropy.constants as const
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import copy


def make_kp_vsys_map(ccf_map, Kp_list, target, normalize=True):
    """
    Function for calculating the Kp-vsys map commonly shown
    in high-res transit papers.
    Parameters
    ----------
    ccf_map: ``redexo.CCF_Dataset``
        CCF in the Earth's(?) rest frame.
    Kp_list: list or np.ndarray
        Array with Kp values to evaluate
    target: ``redexo.Planet``
        Planet to calculate the Kp-vsys map for.
    normalize: bool
        Whether to normalize the Kp-vsys map.
    Returns
    -------
    snr_map: np.ndarray
        Kp-vsys map.
    """
    snr_map = np.zeros((len(Kp_list), ccf_map.rv_grid.shape[-1]))
    mock_target = copy.deepcopy(target)
    for i, kp in enumerate(Kp_list):
        mock_target.Kp = kp
        restframe_ccf_map = ShiftRestFrameModule(
            target=mock_target)(ccf_map.copy())
        flat_ccf = CoAddExposures(
            weights=mock_target.in_transit(ccf_map.obstimes))(
                restframe_ccf_map)
        if normalize:
            flat_ccf.normalize()
        snr_map[i] = flat_ccf.spec[0, 0]
    return snr_map


class ShiftRestFrameModule(Module):
    """
    Module for shifting the restframe of the data.
    """
    def initialise(self, target=None, radial_velocities=None):
        """
        Parameters
        ----------
        target: ``redexo.Planet``
            Planet to which restframe the data is shifted.
        radial_velocities: np.ndarray
            Radial velocities to shift the data to.
        Returns
        -------
        NoneType
            None
        """
        if target is None and radial_velocities is None:
            raise ValueError(("Provide either the target we are observing"
                              "or an array with radial velocities"))
        self.target = target
        self.rvs = radial_velocities

    def process(self, dataset, order_idx, debug=False):
        if self.target is not None:
            self.rvs = -dataset.vbar + self.target.radial_velocity(
                obs_time=dataset.obstimes)

        if isinstance(dataset, CCF_Dataset):
            for exp in range(dataset.num_exposures):
                for order in range(dataset.num_orders):
                    x = dataset.rv_grid[exp, order] - self.rvs[exp]
                    y = dataset.spec[exp, order]
                    dataset.spec[exp, order] = interp1d(
                        x, y, assume_sorted=True,
                        bounds_error=False)(dataset.rv_grid[exp, order])
        else:
            for exp in range(dataset.num_exposures):
                beta = 1-self.rvs[exp]/const.c.to('km/s').value
                new_wl = beta*dataset.wavelengths[exp]
                dataset.spec[exp] = interp1d(
                    new_wl, dataset.spec[exp])(dataset.wavelengths)
        return dataset


class InjectSignalModule(Module):
    """
    Module for injecting a fake signal.
    """
    def initialise(self, template, template_wl, target=None,
                   radial_velocities=None, amplification=1.):
        """
        Parameters
        ----------
        template: np.ndarray
            Model template to use for cross-correlation.
        template_wl: np.ndarray
            Wavelengths at which the template is sampled.
        target: ``redexo.Planet``
            Target which properties are used for the injection.
        radial_velocities: np.ndarray
            Radial velocities of the injected signal.
        amplification: float
            Amplification strength of the injected signal.
        Returns
        -------
        NoneType
            None
        """
        self.template = template
        self.template_wl = template_wl
        self.target = target
        self.rvs = radial_velocities
        self.amplification = amplification

    def process(self, dataset, order_idx, debug=False):
        if self.rvs is None:
            self.rvs = self.target.radial_velocity(obs_time=dataset.obstimes)

        for exp in range(dataset.num_exposures):
            if self.target.in_transit(obs_time=dataset.obstimes[exp]):
                beta = 1-self.rvs[exp]/const.c.to('km/s').value
                wl_new = beta * dataset.wavelengths[exp]
                transit_depth = interp1d(
                    self.template_wl, self.template, bounds_error=True,
                    fill_value=np.median(self.template))(wl_new)
                transit_depth = 1 + self.amplification * (transit_depth - 1)
                dataset.spec[exp] *= transit_depth
        return dataset


class CoAddExposures(Module):
    def initialise(self, indices=None, weights=None):
        self.weights = weights

    def process(self, dataset, order_idx, debug=False):
        if dataset.errors is not None:
            combined_errors = np.sqrt(np.nansum(dataset.errors**2, axis=0))
        if self.weights is None:
            self.weights = np.ones(len(dataset.num_exposures))
        res = np.nansum((self.weights * dataset.spec.T).T, axis=0)
        return dataset.__class__(
            res[np.newaxis, :],
            np.nanmean(dataset.wavelengths, axis=0)[np.newaxis, :],
            combined_errors[np.newaxis, :])


class CoAddOrders(Module):
    def initialise(self, weights=None):
        self.per_order_possible = False
        self.weights = weights
        if isinstance(self.weights, list):
            self.weights = np.array(self.weights)

    def process(self, dataset, order_idx, debug=False):
        if self.weights is not None:
            weights = np.tile(
                self.weights[np.newaxis, :, np.newaxis],
                (dataset.num_exposures, 1, dataset.spec.shape[2]))
            co_added_spec = np.nansum(weights*dataset.spec,
                                      axis=1)[:, np.newaxis]
        else:
            co_added_spec = np.nansum(dataset.spec, axis=1)[:, np.newaxis]
        return dataset.__class__(
            co_added_spec, np.mean(dataset.wavelengths, axis=1)[:, np.newaxis],
            vbar=dataset.vbar, obstimes=dataset.obstimes, *dataset.header_info)


def highpass_gaussian(spec, sigma=100):
    continuum = gaussian_filter1d(spec, sigma)
    return spec/continuum


def broaden(wl, flux, sigma):
    return gaussian_filter1d(flux, sigma)
