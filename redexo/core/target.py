__all__ = ['Planet', 'load_from_exoplanet_archive']
import numpy as np
import astropy.units as u


class Planet(object):
    def __init__(self, orbital_period, Kp=0, vsys=0, T0=0, inclination=90*u.deg, coords = None):
        self.orbital_period = orbital_period
        self.Kp = Kp
        self.vsys = vsys
        self.T0 = T0
        self.inclination = inclination
        self.coords = coords

    @property
    def vsys(self):
        return self._vsys

    @vsys.setter
    def vsys(self, value):
        self._vsys = value

    @property
    def T0(self):
        return self._T0

    @T0.setter
    def T0(self, value):
        self._T0 = value

    @property
    def Kp(self):
        return self._Kp

    @Kp.setter
    def Kp(self, value):
        self._Kp = value
    
    @property
    def inclination(self):
        return self._inclination.to('rad')

    @inclination.setter
    def inclination(self, value):
        self._inclination = value
    
    @property
    def transit_start(self):
        return self._transit_start

    @transit_start.setter
    def transit_start(self, value):
        self._transit_start = value

    def orbital_phase(self, obs_time):
        phase = ((obs_time-self.T0)/self.orbital_period)%1
        phase_around_zero = np.sign(0.5-phase)*np.minimum(phase, np.abs(1-phase))
        return phase_around_zero

    def radial_velocity(self, obs_time=None, phase=None):
        if phase is None and obs_time is None:
            raise ValueError("Provide either the oberving time in bjd or phase of the orbit")
        if phase is not None:
            return self.vsys + self.Kp*np.sin(2*np.pi*phase)*np.sin(self.inclination)
        else:
            return self.vsys + self.Kp*np.sin(2*np.pi*self.orbital_phase(obs_time))*np.sin(self.inclination)

    def in_transit(self, obs_time=None, phase=None):
        if phase is None and obs_time is None:
            raise ValueError("Provide either the observing time in bjd or phase of the orbit")
        if phase is None:
            phase = self.orbital_phase(obs_time)
        return np.abs(phase)<self.transit_start


def load_from_exoplanet_archive(name):
    raise NotImplementedError