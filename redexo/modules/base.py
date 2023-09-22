__all__ = ['Module']


class Module(object):
    def __init__(self, savename=None, per_order=False, **kwargs):
        self.name = savename
        self.per_order = per_order
        self.per_order_possible = True
        self.initialise(**kwargs)

    def initialise(self):
        pass

    def __call__(self, dataset, order_idx=None, debug=False):
        return self.process(dataset, order_idx, debug=debug)

    def process(self, dataset, order_idx, debug=False):
        """
        Runs the Module on the given dataset.
        Parameters
        ----------
        dataset: ``redexo.Dataset``
            Dataset to apply the module to.
        order_idx: int
            Index of the spectral order belonging to the dataset.
        debug: bool
            Wether to print any debug tests.

        Returns
        -------
        result: ``redexo.Datast``
            Processed dataset.
        """
        raise NotImplementedError
