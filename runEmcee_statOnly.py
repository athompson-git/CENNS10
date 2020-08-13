#!/bin/python
#
# runEmcee_statOnly.py
#
# created aug 12, 2020, grayson rich
#
# meant as "simple" adaptation of Adrian Thompson's multinest-based CENNS-10 analysis code
# this uses emcee, rather than multinest

from __future__ import print_function
import numpy as np
from numpy import inf
import scipy.optimize as optimize
from scipy.integrate import ode
from scipy.interpolate import RectBivariateSpline
from scipy.stats import (poisson, norm, lognorm, uniform)
#import scipy.special as special
from scipy.special import gammaln
#from scipy.stats import skewnorm
import sys
import emcee
import csv as csvlib
import argparse
from numbers import Number
from math import isnan
import gc

#
# things used later in sampling
#
outputPrefix = 'emceeOut/'

burninChainFilename = outputPrefix + 'burninchain.txt'
mainChainFilename = outputPrefix + 'mainchain.txt'



burninSteps = 100
mainSteps = 50

useMPI = False
doPlotting = True
#
#
#


if doPlotting == True:
    import matplotlib.pyplot as plt


parNames = [r'CE$\nu$NS', r'N SSBKG', r'N BRN prompt', r'N BRN delayed']


def lnlike(params, observed_data):
    """
    Compute the log-likelihood value
    Note that this DOES NOT account for priors, which are handled separately
    """
    # generate the PDFs with the specified parameter values
    stepPDF = generatePrediction(params)
    bin_by_bin_ll = observed_data * np.log(stepPDF) - stepPDF + gammaln(observed_data + 1)
    return np.sum(bin_by_bin_ll)



def applyPriors(params, priors):
    """
    Returns numerical value reflecting the impact of prior distributions
    params is a list of the parameter values under consideration
    priors is a list of something - either boundaries (uniform priors) and/or PDFs
    
    hardcoding assumptions about priors list for now
    this is definitely not "efficient" or "pythonic"
    """
    priorValue = 0.
    for idx, (parVal,prior) in enumerate(zip(params,priors)):
        # assume each entry in priors is (min range, max range)
        if idx == 0: # case of CEvNS, uniform prior
            if parVal < prior[0] or parVal > prior[1]:
                return -np.inf
        else:
            # all other priors are pdfs of some kind
            priorValue += prior.logpdf(parVal)
    return priorValue

def lnprob(params, priors, observed_data):
    """
    Calculate and return the log probability 
    this combines both priors and likelihood (evidence) or whatever language bayesians use
    """
    logprior = applyPriors(params, priors)
    if not np.isfinite(logprior):
        return -np.inf, -np.inf
    loglike = lnlike(params, observed_data)
    if not np.isfinite(loglike):
        return -np.inf, logprior
    return logprior + loglike, logprior

# Read in data
bkgpdf_data = np.genfromtxt("Data/bkgpdf.txt")
brnpdf_data = np.genfromtxt("Data/brnpdf.txt")
brndelayedpdf_data = np.genfromtxt("Data/delbrnpdf.txt")
cevnspdf_data = np.genfromtxt("Data/cevnspdf.txt")
obs_data = np.genfromtxt("Data/datanobkgsub.txt")

# Set up CEvNS, BRN, and Observed arrays
brn_prompt = brnpdf_data[:,3]
brn_delayed = brndelayedpdf_data[:,3]
obs = obs_data[:,3]
cevns = cevnspdf_data[:,3]
ss = bkgpdf_data[:,3]

# Flat bins
entries = obs_data.shape[0]
keVee = obs_data[:,0]
f90 = obs_data[:,1]
timing = obs_data[:,2]

# define range over which we allow our cevns normalization to float
cevnsRange = (0.1, 3.)

priorPDF_ssBkg_norm = norm.freeze(scale=np.sqrt(np.sum(ss)/5)/np.sum(ss))
priorPDF_BRN_promptNorm = norm.freeze(scale=0.3)

# NOTE: doesn't this readily allow the total number of delayed BRNs to go below 0?
priorPDF_BRN_delayedNorm = norm.freeze(scale=1.0)

priorList = []
priorList.append(cevnsRange)
priorList.append(priorPDF_ssBkg_norm)
priorList.append(priorPDF_BRN_promptNorm)
priorList.append(priorPDF_BRN_delayedNorm)

def generatePrediction(params):
    """
    Makes the 'theoretical' prediction to be used in calc of evidence
    params is a list of the parameter values to be used

    there are hardcoded assumptions about ordering of parameters
    i'm too lazy to provide more detailed description here
    but it should be easy to interpret code
    """
    #TODO: maybe use "in place" numpy arrays? 
    # i think this currently will make new memory spaces for each time
    # the fxn is called
    # this is probably a place where things could be "sped up"
    normalizedPDF_cevns = cevns * params[0]
    normalizedPDF_ssBkg = ss * (1 + params[1])
    normalizedPDF_brnPrompt = brn_prompt * (1 + params[2])
    normalizedPDF_brnDelayed = brn_delayed * (1 + params[3])
    #return normalizedPDF_cevns, normalizedPDF_ssBkg, normalizedPDF_brnPrompt, normalizedPDF_brnDelayed
    return normalizedPDF_cevns + normalizedPDF_ssBkg + normalizedPDF_brnPrompt + normalizedPDF_brnDelayed


# helpful things to store as variables while setting up emcee
nWalkers = 128
nParams = 4

# GENERATE GUESSES (ie starting values of parameters)
# shift them from what we expect as defaults, spread them around a little
#guesses = [np.sum(theHist)*np.random.normal(loc=1.0, scale=0.1) for theHist in [cevns, ss, brn_prompt, brn_delayed]]
guesses = []
guesses.append(1 + np.random.normal(scale=0.1))
for i in range(nParams-1):
    guesses.append(np.random.randn())
    
agitators = [theGuess * 0.15 for theGuess in guesses]
p0 = [guesses + agitators * np.random.randn(nParams) for i in range(nWalkers)]

sampler = emcee.EnsembleSampler(nWalkers, nParams, lnprob, 
                                kwargs={'priors': priorList,
                                'observed_data': obs})

# set up file where we'll dump the chain
fout = open(burninChainFilename,'w')
fout.close()


print('\n\n\nRUNNING BURN IN WITH {0} STEPS\n\n\n'.format(burninSteps))

for i,samplerOut in enumerate(sampler.sample(p0, iterations=burninSteps)):
    if not useMPI or processPool.is_master():
        burninPos, burninProb, priorProb, burninRstate = samplerOut
        print('running burn-in step {0} of {1}...'.format(i, burninSteps))
        fout = open(burninChainFilename, "a")
        for k in range(burninPos.shape[0]):
            fout.write("{0} {1} {2}\n".format(k, burninPos[k], burninProb[k]))
        fout.close()
    
if doPlotting:
    # save an image of the burn in sampling
    
    plt.figure()
    plt.subplot(411)
    plt.plot(sampler.chain[:,:,0].T,'-',color='k',alpha=0.2)
    plt.ylabel(parNames[0])
    plt.subplot(412)
    plt.plot(sampler.chain[:,:,1].T,'-',color='k',alpha=0.2)
    plt.ylabel(parNames[1])
    plt.subplot(413)
    plt.plot(sampler.chain[:,:,2].T,'-', color='k', alpha=0.2)
    plt.ylabel(parNames[2])
    plt.subplot(414)
    plt.plot(sampler.chain[:,:,3].T,'-', color='k', alpha=0.2)
    plt.ylabel(parNames[3])
    plt.xlabel('Burn-in step')
    plt.savefig(outputPrefix + 'emceeBurninSampleChainsOut.png',dpi=300)
    plt.draw()