import numpy as np

#plots
from matplotlib import pyplot as plt
from getdist import plots, MCSamples
import dill
#astrophysics
#import galsim
import xlensing
from astropy.cosmology import FlatLambdaCDM

#saving
from astropy.table import Table
import pickle

#MCMC
import emcee

#utilities
#import os
import time
import warnings
import tqdm

#let's use multiprocessing
from multiprocessing import Pool, freeze_support, cpu_count
from functools import partial

warnings.filterwarnings('ignore')


Zcluster = 0.3
RAcluster = 0.0 #radians 
DECluster = 0.0 #radians
Ngals = 1e6

mratio_list = []
cratio_list = []
mlist = []
clist = []
timestamp = time.time()
if __name__ == '__main__':
  for i in tqdm.tqdm(range(400)):
    logM200 = np.random.uniform(13.5,14.5)
    M200true = 10**logM200
    C200true = xlensing.model.c_DuttonMaccio(Zcluster,M200true)*(np.random.normal(0,0.001)+1)
    print(f"Testing with mass: {M200true: .2e} and concentration {C200true: .2f}. ")

    E1gals, E2gals, RAgals, DECgals, Zgals = xlensing.testing.gen_gal(Ngals=Ngals,Zcluster=Zcluster)


    epsilon = xlensing.testing.NFW_shear( (M200true,C200true, 0, 0, .3),(RAgals, DECgals, Zgals, E1gals, E2gals) )
    e1gals = np.real(epsilon)
    e2gals = np.imag(epsilon)

    e1err = np.array([np.abs(np.random.normal(e/100,np.abs(e/20))) for e in e1gals])
    e2err = np.array([np.abs(np.random.normal(e/100,np.abs(e/20))) for e in e2gals])
    Wgals = (0.1**2 + e1err**2 +e2err**2)/(0.1**2 + e1err**2 +e2err**2) #w=1
    Mgals = -np.random.exponential(0.03,size=int(Ngals))*0

    galaxy_catalog = Table([RAgals,DECgals,Zgals,e1gals,e2gals, Wgals,Mgals],names=['RA','DEC','ZPHOT','E1','E2','WEIGHT','M'])


    sr_RA = np.array(galaxy_catalog['RA'])
    sr_DEC= np.array(galaxy_catalog['DEC'])
    sr_z  = np.array(galaxy_catalog['ZPHOT'])
    sr_E1 = np.array(galaxy_catalog['E1'])
    sr_E2 = np.array(galaxy_catalog['E2'])
    sr_W = np.array(galaxy_catalog['WEIGHT'])
    sr_M = np.array(galaxy_catalog['M'])


    clusters = Table([[RAcluster],[DECluster],[Zcluster]],names=['RA','DEC', 'Z'])
    clusters['INDEX'] = np.array(range(len(clusters)))

    pool = Pool(cpu_count()) 


    #We get a partial function with a constant galaxy catalogue to iterate with clusters.

    survey_lensing = partial(xlensing.data.cluster_lensing,sources=(sr_RA, 
                                                                    sr_DEC, 
                                                                    sr_z, 
                                                                    sr_E1, 
                                                                    sr_E2, 
                                                                    sr_W,
                                                                    sr_M),radius=10.)

    #Make a list of clusters to get lensing data
    cl_RA=np.array(clusters['RA'])
    cl_DEC= np.array(clusters['DEC'])
    cl_z= np.array(clusters['Z'])
    cl = np.array([cl_RA,cl_DEC,cl_z]).T
    clz = zip(cl_RA,cl_DEC,cl_z)
    clzlist = [x for x in clz]

    results = pool.map(survey_lensing, clzlist)

    stick = [clusters]


    radii = np.logspace(-0.8,0.8,8)
    N = len(radii)
    bins_lims = np.logspace(np.log10(radii[0])+(np.log10(radii[0])-np.log10(radii[1]))/2,
                            np.log10(radii[N-1])-(np.log10(radii[0])-np.log10(radii[1]))/2,N+1)
    bins_lims = np.array([[bins_lims[i],bins_lims[i+1]] for i in range(N)])


    Nboot=200
    stick_results = []
    for stake in stick:
        t = time.time()
        clusterbkgs = []
        for index in stake['INDEX']:
            Sigma_crit = np.array(results[index]['Critical Density'])
            e_t = np.array(results[index]['Tangential Shear'])
            e_x = np.array(results[index]['Cross Shear'])
            W = np.array(results[index]['Weights'])
            M = np.array(results[index]['Mult. Bias'])
            R = np.array(results[index]['Radial Distance'])
            clusterbkgs.append(np.array([Sigma_crit, e_t, e_x, W, R,M]))
        print(len(clusterbkgs))
        sigmas, sigmas_cov, xigmas, xigmas_cov = xlensing.data.stack(clusterbkgs,bins_lims,Nboot)
        stick_results.append( ( sigmas, sigmas_cov, xigmas, xigmas_cov) )
        print("Done in " + str(time.time()-t) + " seconds.")

    def NFWsimple(theta,Z,radii):
      logM200, C200  = theta
      M200 = np.power(10,logM200)
      result = xlensing.model.NFW_shear(M200, C200, Z, 1.0, 0.001, 1e10,radii)['NFW Signal'] #returns only the main shear signal - all other signals (incl cross signal) available see docstring
      return result

    M200lo, M200hi = 13, 15
    C200lo, C200hi = 0, 10

    priorM200 = xlensing.fitting.ln_flat_prior_maker(M200lo, M200hi,0)
    priorC200 = xlensing.fitting.ln_flat_prior_maker(C200lo, C200hi,1)
    #priorPCC = xlensing.fitting.ln_gaussian_prior_maker(0.75, 0.07,2) ##Zhang et al. 2019
    prior = lambda theta : priorM200(theta) + priorC200(theta)# + priorPCC(theta)

    ndim, nwalkers, steps = 2, 256, 256
    samplestick = []
    #for each stack, run MCMC
    burnin=round(steps/4.)
    for stickresult in stick_results:
      mean_z = Zcluster

      #build data likelihood
      model = lambda theta: NFWsimple(theta,mean_z,radii)
      likelihood = xlensing.fitting.ln_gaussian_likelihood_maker((stickresult[0],stickresult[1]),model)
      posterior = lambda theta : likelihood(theta) +prior(theta)

      #initialise walkers
      pos = []
      for i in range(nwalkers):
          M200 = np.random.uniform(M200lo,M200hi)
          C200 = np.random.uniform(C200lo,C200hi)
          #PCC  = np.random.uniform(PCClo,PCChi)
          pos.append(np.array([M200,C200]))
      pool = Pool(cpu_count()) 
      sampler = emcee.EnsembleSampler(nwalkers, ndim, posterior,threads=48)
      print("Running MCMC...")
      t = time.time()
      sampler.run_mcmc(pos, steps, rstate0=np.random.get_state())
      print("Done in " + str(time.time()-t) + " seconds.")
      samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
      samplestick.append(samples)
    for samples in samplestick:
      mvir_tru,conc_tru= map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), 
                             zip(*np.percentile(samples, [16, 50, 84],axis=0)))
      #print("Mvir: {:.2e}".format(mvir_tru[0]) + " p {:.2e}".format(mvir_tru[1]) + " m {:.2e}".format(mvir_tru[2]))
      #print("Conc: {:.2f}".format(conc_tru[0]) + " p {:.2f}".format(conc_tru[1]) + " m {:.2f}".format(conc_tru[2]))

    m_ratio = 10**mvir_tru[0]/M200true
    c_ratio = conc_tru[0]/C200true
    print(m_ratio)
    print(c_ratio)
    mratio_list.append(m_ratio)
    cratio_list.append(c_ratio)
    mlist.append(M200true)
    clist.append(C200true)
    np.save(f"mtrue_mass_range_higher_snr{timestamp}.npy",np.array(mlist))
    np.save(f"ctrue_mass_range_higher_snr{timestamp}.npy",np.array(clist))
    np.save(f"mratio_mass_range_higher_snr{timestamp}.npy",np.array(mratio_list))
    np.save(f"cratio_mass_range_higher_snr{timestamp}.npy",np.array(cratio_list))

