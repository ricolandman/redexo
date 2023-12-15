#------ manually fix import for now
import sys
sys.path.append('..')
#------

import numpy as np
import astropy.units as u
import astropy.constants as const
import emcee
from redexo import *
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import copy
import corner

dataset = load_carmenes_data(folder='/users/ricolandman/Documents/Research/OH_exoplanets/Data/Wasp76_NIR/',\
                            spectral_orders=[23,24,25,26,27])
planet = Planet(Kp=196, vsys=-1.1, T0=2458080.626165, orbital_period=1.809886)
planet.transit_start = 0.045

pipeline = Pipeline()

pipeline.add_module( FlagAbsorptionEmissionModule(flux_lower_limit=0.4, flux_upper_limit=1.1, 
                                                    relative_to_continuum=True) )
pipeline.add_module( PolynomialContinuumRemovalModule(poly_order=3))


pipeline.add_module( OutlierFlaggingModule(sigma=5) )

pipeline.add_module( SysRemModule(number_of_modes=6, mode='subtract', name='cleaned') )


pipeline.run(dataset, num_workers=7, per_order=True) 
pipeline.summary()

res = pipeline.get_results('cleaned')

def likelihood_gibson(template, data, errors=None):
    if not errors is None:
        chi2 = np.nansum((template-data)**2/errors**2)
    else:
        chi2 = np.nansum((template-data**2))
    '''
    #template -= np.mean(template)
    #data -= np.mean(data)
    R = np.mean(template*data)
    sf2 = np.mean(data**2)
    sg2 = np.std(template**2)
    logL = -template.size/2 *np.log(sf2 - 2*R +sg2)
    '''
    return -template.size/2*np.log(chi2)


def likelihood_brogi_line(template, data):
    #template -= np.mean(template)
    #data -= np.mean(data)
    R = np.nanmean(template*data)
    sf2 = np.nanmean(data**2)
    sg2 = np.nanstd(template**2)
    logL = -template.size/2 *np.log(sf2 - 2*R +sg2)
    return logL

class LikelihoodEvaluator():
    def __init__(self, dataset, template_dict, target):
        self.dataset = dataset
        
        wl_arr = dataset.wavelengths.reshape(dataset.num_exposures, -1)
        flux_arr = dataset.spec.reshape(dataset.num_exposures, -1)+1
        error_arr = dataset.errors.reshape(dataset.num_exposures, -1)
        
        self.wls = [wl_arr[i][~np.isnan(flux_arr[i])] for i in range(self.dataset.num_exposures)]
        self.errors = [error_arr[i][~np.isnan(flux_arr[i])] for i in range(self.dataset.num_exposures)]
        self.fluxes = [flux_arr[i][~np.isnan(flux_arr[i])] for i in range(self.dataset.num_exposures)]
        self.template_dict = template_dict
        modelWave, modelTrans = self.template_dict['2200']
        model_res = np.median(modelWave[:-1]/np.diff(modelWave))
        self.model_dv = const.c.to('km/s').value/model_res
        self.target = target

    def __call__(self, parameters):
        return self.evaluate(self.parameter_dict(parameters))

    def evaluate(self, parameters):
        if parameters['Kp']<100 or parameters['Kp']>300:
            return -np.inf
        elif np.abs(parameters['vsys'])>40:
            return -np.inf
        elif parameters['Temperature']<1500 or parameters['Temperature']>3500:
            return -np.inf
        elif parameters['fwhm']<=0 or parameters['fwhm']>50:
            return -np.inf
        elif parameters['alpha']<0 or parameters['alpha']>5:
            return -np.inf
        self.target.Kp = parameters['Kp']
        self.target.vsys = parameters['vsys']
        T = 100*int(np.round(parameters['Temperature']/100))
        modelWave, modelTrans = self.template_dict[str(T)]
        sigma = (parameters['fwhm']/self.model_dv)/ (2*np.sqrt(2. *np.log(2.)))
        modelTrans = parameters['alpha']*broaden(modelWave, modelTrans, sigma)
        rvs = -self.dataset.vbar+self.target.radial_velocity(self.dataset.obstimes)
        logL = 0
        for exp, rv in enumerate(rvs):
            beta = 1.-rv/const.c.to('km/s').value
            shifted_wavelengths = beta*self.wls[exp]
            if np.min(shifted_wavelengths)<np.min(modelWave):
                return -np.inf
            T = interp1d(modelWave, modelTrans, bounds_error=True, fill_value=0)(shifted_wavelengths)
            #T -= np.mean(T)
            #logL += likelihood_brogi_line(T, self.fluxes[exp])
            logL += likelihood_gibson(T, self.fluxes[exp], self.errors[exp])
        return logL

    def parameter_dict(self, param_array):
        param_dict = {}
        param_dict['Kp']= param_array[0]
        param_dict['vsys']= param_array[1]
        param_dict['Temperature']= param_array[2]
        param_dict['fwhm']= param_array[3]
        param_dict['alpha'] = param_array[4]
        return param_dict

print('Reading templates')
template_dict = {}
for temp in np.arange(1500, 3600, 50):
    modelWave, modelTrans = np.load('/users/ricolandman/Documents/Research/OH_exoplanets/Data/models/Wasp76_OH_equilibrium_T={0}K_prepared.npy'.format(temp))
    template_dict[str(temp)] = (modelWave, modelTrans)

print('Preparing likelihood evaluator')
likelihood_evaluator = LikelihoodEvaluator(res, template_dict, planet)

ndim, nwalkers = 5, 50
Kp0 = 230 + np.random.randn(nwalkers)
vsys0 = -10 + np.random.randn(nwalkers)
temp0 = 2200+ 100*np.random.randn(nwalkers)
fwhm0 = 1+ 0.1*np.random.randn(nwalkers)
alpha0 = 1+ 0.1*np.random.randn(nwalkers)
p0 = np.concatenate([arr[np.newaxis,:] for arr in [Kp0, vsys0, temp0, fwhm0,alpha0]], axis=0).T

sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood_evaluator)
sampler.run_mcmc(p0, 1000, progress=True)

burnin=500
samples = sampler.get_chain(discard=burnin,thin=2).reshape(-1,5)

quantiles = [ 0.5,0.26, 0.84]
labels= ['Kp', 'vsys','Temperature', 'FWHM','alpha']
for i in range(5):
    med = np.median(samples[:,i])
    q1 = np.quantile(samples[:,i], 0.16)
    q2 = np.quantile(samples[:,i], 0.84)
    print(labels[i], ': {0:.2f}  {1:.2f}  {2:.2f}'.format(med,q1-med, q2-med))



clean_labels = ['K_p', 'v_{rest}', 'T','FWHM', r'\alpha']
units = [' km/s',' km/s',' K',' km/s','']
title_fmt = ['.1f',".1f",".0f",".1f",".2f"]
titles = []
for i in range(5):
    med = np.median(samples[:,i])
    med = np.median(samples[:,i])
    q1 = np.quantile(samples[:,i], 0.16)
    q2 = np.quantile(samples[:,i], 0.84)
    fmt = "{{0:{0}}}".format(title_fmt[i]).format
    #title = r'{0} = {1:.2f}^{+{2:.2f}}'.format(clean_labels[i],med,q2-med)
    title = r"${0}={{{1}}}_{{-{2}}}^{{+{3}}}${4}".format(clean_labels[i], fmt(med),
                                                       fmt(med-q1),fmt(q2-med),units[i])
    #print(title)
    titles.append(title)
    #title = title.format(fmt(q_50), fmt(q_m), fmt(q_p))
    #print(title)
    print(labels[i], ': {0:.2f}  {1:.2f}  {2:.2f}'.format(med,q1-med, q2-med))
print(titles)

labels = [r'$K_p$ [km/s]', r'$v_{rest}$ [km/s]', 'T [K]','FWHM [km/s]', r'$\alpha$']
print(titles)
figure = corner.corner(samples,labels=labels, label_kwargs={'fontsize':14},
              title_kwargs={'fontsize':13},titles=titles,show_titles=True,title_fmt=None, quantiles=[0.16, 0.5, 0.86])
axes = np.array(figure.axes).reshape((5,5))
plt.show()