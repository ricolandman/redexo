__all__ = ['OutlierFlaggingModule', 'PolynomialContinuumRemovalModule', 'FlagAbsorptionEmissionModule']
from .base import Module
import numpy as np
import matplotlib.pyplot as plt


class FlagAbsorptionEmissionModule(Module):
    '''
    Flags parts of the spectra that have less flux than flux_lower_limit or more flux than flux_upper_limit
    '''
    def initialise(self, flux_lower_limit=0, flux_upper_limit=np.inf, relative_to_continuum=False):
        self.flux_lower_limit = flux_lower_limit
        self.flux_upper_limit = flux_upper_limit
        self.relative_to_continuum = relative_to_continuum

    def process(self, dataset, debug=False):
        if self.relative_to_continuum:
            continuum_removed_data = PolynomialContinuumRemovalModule(poly_order=3)(dataset.copy())
            spec_norm = np.nanmedian(continuum_removed_data.spec, axis=0)
        else:
            spec_norm = np.nanmedian(dataset.spec, axis=0)
        mask = (spec_norm>self.flux_lower_limit)*(spec_norm<self.flux_upper_limit)
        if debug:
            print('Masking {0:.2f}% of the data'.format(np.sum(~mask)/mask.size*100))
        dataset.spec[:,~mask] = np.nan
        return dataset

class OutlierFlaggingModule(Module):
    def initialise(self, sigma=5):
        self.sigma = sigma

    def process(self, dataset, debug=False):
        std = np.nanstd(dataset.spec)
        mean = np.nanmean(dataset.spec)
        outliers = (np.abs(dataset.spec - mean)/std)>self.sigma
        if debug:
            print('Masking {0:.2f}% of the data'.format(np.sum(outliers)/outliers.size*100))
        dataset.spec[outliers] = np.nan
        return dataset


class PolynomialContinuumRemovalModule(Module):
    def initialise(self, poly_order=3):
        self.poly_order = poly_order

    def process(self, dataset, debug=False):
        for exp in range(dataset.num_exposures):
            nans = np.isnan(dataset.spec[exp])
            cont_model = np.poly1d(np.polyfit(dataset.wavelengths[exp][~nans], dataset.spec[exp][~nans], self.poly_order))
            continuum = cont_model(dataset.wavelengths[exp][~nans])
            dataset.spec[exp][~nans] = dataset.spec[exp][~nans]/continuum
            dataset.errors[exp][~nans] = dataset.errors[exp][~nans]/continuum
        return dataset