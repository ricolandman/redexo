__all__ = ['Dataset', 'CCF_Dataset']
import numpy as np
import astropy.units as u
import copy


class Dataset(object):
    """
    Main dataset class that keeps the spectra, errors and wavelengths
    """

    def __init__(self, spec=None, wavelengths=None, errors=None, vbar=None,
                 obstimes=None, **kwargs):
        """
        Parameters
        ----------
        spec : np.ndarray
            Array with spectra, should have dimensions:
            (exposures, orders, wavelengths). Can also be None and iteratively
            added using `add_exposure`.
        wavelengths: np.ndarray
            Array with wavelength bins, should have dimensions:
            (exposures, orders, wavelengths). Can also be None and iteratively 
            added using `add_exposure`.
        errors: np.ndarray
            Array with errors, should have dimensions:
            (exposures, orders, wavelengths). Can also be None and iteratively
            added using `add_exposure`.
        vbar: list
            List with barycentric velocity corrections.
        obstimes: list
            List with observing times.
        kwargs: dict
            Use this when you want to store more data regarding the
            exposures, e.g. airmass or signal-to-noise ratio.
            This will be stored in the `header_info` dictionary.

        Returns
        -------
        NoneType
            None
        """

        self.header_info = dict()
        for key, value in kwargs.items():
            self.header_info[key] = value
        if vbar is None:
            vbar = []
        if obstimes is None:
            obstimes = []
        self._vbar = vbar
        self._obstimes = obstimes
        if spec is not None:
            if spec.ndim == 2:
                self.spec = spec[:, np.newaxis, :] 
            else:
                self.spec = spec

            if wavelengths.ndim == 1:
                self.wavelengths = np.tile(
                    wavelengths[np.newaxis, np.newaxis, :],
                    (self.num_exposures, self.num_orders, 1))
            elif wavelengths.ndim == 2:
                self.wavelengths = np.tile(
                    wavelengths[np.newaxis, :, :], 
                    (self.num_exposures, 1, 1))
            else:
                self.wavelengths = wavelengths
            if errors is not None:
                self.errors = errors
            else:
                self.errors = np.ones_like(self.spec)

    @property
    def num_orders(self):
        return self.spec.shape[1]

    @property
    def num_exposures(self):
        return self.spec.shape[0]

    @property
    def phases(self):
        return self.target.orbital_phase(self.obstimes)

    @property
    def transit_indices(self):
        return np.where(self.target.in_transit(self.phases))[0]

    @property
    def planet_rvs(self):
        return (-self.vbar * u.km/u.s) + self.target.planet_rv(self.phases)

    @property
    def obstimes(self):
        return np.array(self._obstimes)

    @property
    def vbar(self):
        return np.array(self._vbar)
    
    def add_exposure(self, spectrum, wl, errors, vbar=None,
                     obstime=None, **kwargs):
        """
        Method for adding data from an exposure to the dataset.

        Parameters
        ----------
        spectrum: np.ndarray
            Array with the spectrum to add to the dataset.
        wl: np.ndarray
            Array with the wavelengths to add to the dataset.
        errors: np.ndarray
            Array with the errors to add to the dataset.
        vbar: float
            Barycentric velocity correction for the added frame.
        obstime: float
            Time of the observation in MJD.
        kwargs: dict
            Use this when you want to store more data regarding the
            exposures, e.g. airmass or signal-to-noise ratio.
            This will be stored in the `header_info` dictionary.

        Returns
        -------
        NoneType
            None
        """
        if not hasattr(self, 'spec'):
            assert spectrum.ndim == 2,\
                ("Flux should have shape:"
                 "(num_orders, wavelength_bins_per_order)")
            self.spec = spectrum[np.newaxis, :]
            self.wavelengths = wl[np.newaxis, :]
            if errors is not None:
                self.errors = errors[np.newaxis, :]
            else:
                self.errors = np.ones_like(self.spec)
        else:
            assert spectrum.shape == self.spec.shape[1:],\
                ("Added flux should have shape (num_orders, "
                 "wavelength_bins_per_order)")
            self.spec = np.concatenate([self.spec, spectrum[np.newaxis, :]],
                                       axis=0)
            self.wavelengths = np.concatenate(
                [self.wavelengths, wl[np.newaxis, :]], axis=0)
            if errors is not None:
                self.errors = np.concatenate(
                    [self.errors, errors[np.newaxis, :]], axis=0)

        if obstime is not None:
            self._obstimes.append(obstime)
        if vbar is not None:
            self._vbar.append(vbar)

        for key, value in kwargs.items():
            self.header_info[key].append(value)

    def drop_exposures(self, drop_indices):
        """
        Method to drop exposures.

        Parameters
        ----------
        drop_indices: list
            List with indices of the exposures to drop.

        Returns
        -------
        NoneType
            None
        """
        to_not_drop = [idx for idx in np.arange(self.num_exposures) if
                       idx not in drop_indices]
        self.spec = self.spec[to_not_drop]
        self.wavelengths = self.wavelengths[to_not_drop]
        self.errors = self.errors[to_not_drop]

        self._vbar = [self._vbar[idx] for idx in to_not_drop]
        self._obstimes = [self._obstimes[idx] for idx in to_not_drop]

    def set_results(self, spec, wavelengths=None, errors=None, order=None):
        """
        Method to set the results of an operation on the dataset.

        Parameters
        ----------
        spec: np.ndarray
            New values of the spectrum.
        wavelengths: np.ndarray
            New values of the wavelengths.
        errors: np.ndarray
            New values of the errors.
        order: int
            Use this if you only want to set the results of
            a specific spectral order.

        Returns
        -------
        NoneType
            None
        """
        if isinstance(spec, Dataset) or isinstance(spec, CCF_Dataset):
            wavelengths = spec.wavelengths
            errors = spec.errors
            spec = spec.spec
        else:
            if order is None:
                if not spec.shape == self.spec.shape:
                    raise ValueError(("The spec you are trying to set has"
                                      "an incorrect shape, expected:"
                                      "{0}, received: {1}"), self.spec.shape,
                                     spec.shape)
                order = np.arange(self.num_orders)

        self.spec[:, order:order+1, :] = spec
        if wavelengths is not None:
            self.wavelengths[:, order:order+1, :] = wavelengths
        if errors is not None:
            self.errors[:, order:order+1, :] = errors

    def get_order(self, order, as_dataset=True):
        """
        Method to obtain the information from one spectral order.

        Parameters
        ----------
        order: int
            Spectral order to obtain from the dataset.
        as_dataset: bool
            Return the spectrum as a ``Dataset`` object.
            If ``False`` the spectrum, wavelengths and errors will
            be returned as numpy arrays.

        Returns
        -------
        NoneType
            None
        """
        if not as_dataset:
            return self.spec[:, order], self.wavelengths[:, order],\
                self.errors[:, order]
        else:
            return self.__class__(self.spec[:, order, np.newaxis],
                                  self.wavelengths[:, order, np.newaxis],
                                  self.errors[:, order, np.newaxis],
                                  self.vbar, self.obstimes, *self.header_info)

    def make_clean(self, shape):
        """
        Method to re-initialize the ``Dataset`` object.

        Parameters
        ----------
        shape: tuple
            Shape to initialize the arrays to.

        Returns
        -------
        NoneType
            None
        """
        self.spec = np.zeros(shape)
        self.wavelengths = np.zeros(shape)
        self.errors = np.zeros(shape)
        self.vbar = []
        self.obstimes = []
        self.header_info = {}

    def copy(self):
        return copy.deepcopy(self)


class CCF_Dataset(Dataset):
    """
    Class that keeps the CCF values and radial velocities.
    """
    def __init__(self, spec=None, rv_grid=None, vbar=[], obstimes=[],
                 **kwargs):
        super().__init__(spec=spec, wavelengths=rv_grid, vbar=vbar,
                         obstimes=obstimes, **kwargs)

    @property
    def rv_grid(self):
        return self.wavelengths

    @rv_grid.setter
    def rv_grid(self, rv_grid):
        self.wavelengths = rv_grid

    def get_order(self, order, as_dataset=True):
        """
        Method to obtain the information from one spectral order.

        Parameters
        ----------
        order: int
            Spectral order to obtain from the dataset.
        as_dataset: bool
            Return the spectrum as a ``Dataset`` object.
            If ``False`` the CCF and radial velocities will
            be returned as numpy arrays.

        Returns
        -------
        NoneType
            None
        """
        if not as_dataset:
            return self.spec[:, order], self.rv_grid[:, order],\
                self.errors[:, order]
        else:
            return CCF_Dataset(self.spec[:, order, np.newaxis],
                               self.rv_grid[:, order, np.newaxis],
                               self.vbar, self.obstimes, *self.header_info)
        
    def normalize(self, exclude_region=50):
        """
        Method to normalize CCF maps

        Parameters
        ----------
        exclude_region: float
            Only include radial velocities (in km/s)
            larger than exclude_region for the normalization

        Returns
        -------
        NoneType
            None
        """
        mask = np.abs(self.rv_grid[0, 0]) > exclude_region
        means = np.tile(np.nanmean(
            self.spec[:, :, mask], axis=-1)[:, :, np.newaxis],
            (1, 1, self.spec.shape[-1]))
        stds = np.tile(np.nanstd(
            self.spec[:, :, mask], axis=-1)[:, :, np.newaxis],
            (1, 1, self.spec.shape[-1]))
        self.spec = (self.spec-means)/stds
        return self
