__all__ = ['Pipeline']

import numpy as np
from astropy.io import fits
import copy
from pathos.multiprocessing import ProcessingPool as Pool
from .dataset import Dataset, CCF_Dataset
import time


class Pipeline(object):
    """
    Class for defining and running the reduction pipeline.
    """
    def __init__(self, in_memory=True):
        """
        Parameters
        ----------
        in_memory: bool
            Wether to keep the result from the module is memory.

        Returns
        -------
        NoneType
            None
        """
        self.modules = []
        self.database = {}
        self.in_memory = in_memory
        if in_memory is False:
            raise NotImplementedError()

    def add_module(self, module):
        """
        Method for adding a module to the pipeline.
        Parameters
        ----------
        module : ``redexo.Module``
            Module to add to the reduction pipeline.

        Returns
        -------
        NoneType
            None
        """

        self.modules.append(module)

    def run(self, dataset, per_order=True, num_workers=None, debug=False,
            start=None, end=None):
        """
        Method for running the reduction pipeline.
        Parameters
        ----------
        dataset : ``redexo.Dataset``
            Dataset to reduce.
        per_order : bool
            Wether to reduce the dataset per order if possible.
            This allows for parallelization.
        num_workers : int
            Number of parallel processes to use for the data reduction.
        start : str
            Name of the module to start the reduction pipeline from.
        end : str
            Name of the module to stop the reduction pipeline after.

        Returns
        -------
        NoneType
            None
        """
        processed_dataset = dataset.copy()
        self.run_times = []
        self.module_names = [module.name for module in self.modules]

        # Find where to continue running the pipeline and where to end
        if start is not None:
            if start not in self.module_names:
                raise ValueError("Module name {0} not found in pipeline".
                                 format(start))
            start_idx = np.where([name == start for
                                  name in self.module_names])[0][0]
            if not self.module_names[start_idx-1] in self.database.keys():
                raise ValueError(("Modules before start module have not"
                                  "yet been run"))
            processed_dataset = self.get_results(
                self.module_names[start_idx - 1])
        else:
            start_idx = 0
        
        if end is not None:
            if end not in self.module_names:
                raise ValueError("Module name {0} not found in pipeline"
                                 .format(end))
            end_idx = np.where([name == end for name
                                in self.module_names])[0][0] + 1

        if end is not None:
            modules_to_run = self.modules[start_idx:end_idx]
        else:
            modules_to_run = self.modules[start_idx:]

        # Propagate through the modules
        for module in modules_to_run:
            t = time.time()
            if (per_order or module.per_order) and module.per_order_possible:

                if num_workers is not None:
                    with Pool(num_workers) as p:
                        results = p.map(lambda order: module.process(
                            processed_dataset.get_order(order), order,
                            debug=debug), range(processed_dataset.num_orders))
                else:
                    results = [module.process(
                        processed_dataset.get_order(order), order, debug=debug)
                        for order in range(processed_dataset.num_orders)]

                # Check if resulting dataset still matches our previous dataset
                # If not, make sure they get compatible
                if isinstance(results[0], CCF_Dataset) and not \
                   isinstance(processed_dataset, CCF_Dataset):
                    empty_spec = np.zeros((results[0].spec.shape[0],
                                           processed_dataset.num_orders,
                                           results[0].spec.shape[-1]))
                    processed_dataset = CCF_Dataset(
                        empty_spec, rv_grid=empty_spec.copy(),
                        vbar=processed_dataset.vbar,
                        obstimes=processed_dataset.obstimes,
                        **dataset.header_info)
                elif results[0].spec.shape != processed_dataset.spec.shape:
                    processed_dataset.make_clean(
                        shape=(results[0].spec.shape[0],
                               processed_dataset.num_orders,
                               results[0].spec.shape[-1]))

                # Write results to dataset
                for order, res in enumerate(results):
                    processed_dataset.set_results(res, order=order)
            else:
                processed_dataset = module.process(
                    processed_dataset.copy(), order_idx=None, debug=debug)
            if module.name is not None:
                self.database[module.name] = processed_dataset.copy()
            self.run_times.append(time.time() - t)

    def get_results(self, name):
        """
        Method for getting the results from a module.
        Parameters
        ----------
        name: str
            Name of the module to get the results from.

        Returns
        -------
        result: ``redexo.Dataset``
            The result of the processing module.
        """
        return self.database[name]

    def write_results(self, filepath, modules=None):
        """
        Method for writing the results  from modules to 
        Parameters
        ----------
        name: str
            Name of the module to get the results from.

        Returns
        -------
        result: ``redexo.Dataset``
            The result of the processing module.
        """
        self.hdulist = fits.HDUList()
        self.hdulist.append(fits.PrimaryHDU())
        for savename, dataset in self.database.items():
            if isinstance(dataset, CCF_Dataset):
                self.hdulist.append(fits.ImageHDU(
                    dataset.spec, name=savename + "_CCF"))
                self.hdulist.append(fits.ImageHDU(
                    dataset.rv_grid, name=savename + "_radial_velocities"))
            else:
                self.hdulist.append(fits.ImageHDU(
                    dataset.spec, name=savename + "_spec"))
                self.hdulist.append(fits.ImageHDU(
                    dataset.wavelengths, name=savename + "_wavelengths"))
                self.hdulist.append(fits.ImageHDU(
                    dataset.errors, name=savename + "_errors"))
        self.hdulist.writeto(filepath)

    def summary(self):
        """
        Method for printing the summary of the reduction.
        """
        print('----------Summary--------')
        for i, module in enumerate(self.modules):
            print('Running {0} took {1:.2f} seconds'.format(
                module, self.run_times[i]))
        print('--> Total time: {0:.2f} seconds'.format(
            np.sum(np.array(self.run_times))))
        print('------------------------')
