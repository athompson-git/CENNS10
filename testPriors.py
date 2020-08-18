from numpy import *
import numpy as np
from scipy.special import erfinv, gammaln
from scipy.stats import skewnorm, norm, truncnorm
import matplotlib.pyplot as plt

import pymultinest
from pymultinest import Analyzer
import mpi4py

import json
import time

# Read in data
bkgpdf_data = genfromtxt("Data/bkgpdf.txt")
brnpdf_data = genfromtxt("Data/brnpdf.txt")
brndelayedpdf_data = genfromtxt("Data/delbrnpdf.txt")
cevnspdf_data = genfromtxt("Data/cevnspdf.txt")
obs_data = genfromtxt("Data/datanobkgsub.txt")

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


# Define stats CDFs for priors
ss_error = sqrt(sum(ss)/5)/sum(ss) # percent error
normSS = norm(scale=ss_error) #truncnorm(-1.0,1.0,scale=ss_error)
normPromptBRN = norm(scale=0.3) #truncnorm(-1.0,1.0,scale=0.3)
normDelayedBRN = norm(scale=1.0) #truncnorm(-1.0,1.0,scale=1.0)




# Define Priors for MultiNest
def prior_stat(cube, n, d):
    cube[0] = 2*cube[0]  # CEvNS norm
    cube[1] = normSS.ppf(cube[1])  # SS norm
    cube[2] = normPromptBRN.ppf(cube[2])  # BRN prompt norm
    cube[3] = normDelayedBRN.ppf(cube[3])  # BRN delayed norm

def prior_stat_null(cube, n, d):
    cube[0] = normSS.ppf(cube[0])  # SS norm
    cube[1] = normPromptBRN.ppf(cube[1])  # BRN prompt norm
    cube[2] = normDelayedBRN.ppf(cube[2])  # BRN delayed norm 




# Generate new PDFs with nuisance-controlled norms
def events_gen_stat(cube, report_stats=False):
    brn_prompt_syst = ((1+cube[2])*brn_prompt).clip(min=0.0001)
    brn_del_syst = ((1+cube[3])*brn_delayed).clip(min=0.0001)
    cevns_syst = (cube[0]*cevns).clip(min=0.0001)
    ss_syst = ((1+cube[1])*ss).clip(min=0.0001)

    if report_stats:
        print("N_CEvNS = ", sum(cevns_syst))
        print("N_BRN_PRO = ", sum(brn_prompt_syst))
        print("N_BRN_DEL = ", sum(brn_del_syst))
        print("N_SS = ", sum(ss_syst))

    return brn_prompt_syst + brn_del_syst + cevns_syst + ss_syst

# Generate new PDFs with nuisance-controlled norms (no CEvNS)
def events_gen_stat_null(cube):
    brn_prompt_syst = ((1+cube[1])*brn_prompt).clip(min=0.0001)
    brn_del_syst = ((1+cube[2])*brn_delayed).clip(min=0.0001)
    ss_syst = ((1+cube[0])*ss).clip(min=0.0001)

    return brn_prompt_syst + brn_del_syst + ss_syst




def poisson(obs, theory):
    ll = obs * log(theory) - theory - gammaln(obs+1)
    return sum(ll)

def PrintSignificance():
    # Print out totals.
    print("TOTALS:")
    print("N_obs = ", sum(obs))
    print("N_ss = ", sum(ss))
    print("N_brn =", sum(brn_prompt + brn_delayed))
    print("N_cevns = ", sum(cevns))

    an = Analyzer(4, "multinest/cenns10_stat/cenns10_stat")
    bf = an.get_best_fit()['parameters']

    an_null = Analyzer(3, "multinest/cenns10_stat_no_cevns/cenns10_stat_no_cevns")
    bf_null = an_null.get_best_fit()['parameters']
    bf = [an.get_stats()['marginals'][0]['median'], an.get_stats()['marginals'][1]['median'],
          an.get_stats()['marginals'][2]['median'], an.get_stats()['marginals'][3]['median']]
    bf_null = [an_null.get_stats()['marginals'][0]['median'], an_null.get_stats()['marginals'][1]['median'],
               an_null.get_stats()['marginals'][2]['median']]
    # Save best-fit (MLE) parameters from MultiNest (in <out>stats.dat)
    # Truncated gaussian
    bf_norm = [0.128203949389575733E+01,
              -0.757751720547599188E-02,
               0.928830540200280969E-01,
              -0.681121212215910043E+00]
    bf_norm_null = [-0.799580130637969101E-02,
                     0.253213583049654078E+00,
                    -0.514351228113789194E+00]
    # Unconstrained Gaussian
    bf_truncnorm = [0.168960153287222759E+01,
                   -0.312937517992761469E-01,
                    0.780942325684447630E-01,
                   -0.970385882467374672E+00]
    bf_truncnorm_null = [-0.147905160635425168E-01,
                          0.245574237324468231E+00,
                         -0.460897530294036739E+00]

    # Get ratio test
    print("Significance (stat):")
    stat_q = sqrt(abs(2*(-poisson(obs, events_gen_stat(bf)) \
                        + poisson(obs, events_gen_stat_null(bf_null)))))
    print(stat_q)

    print("Best-fit norms:")
    events_gen_stat(bf, report_stats=True)




def RunMultinest():
    def loglike(cube, ndim, nparams):
        return np.random.uniform(-1,0)

    save_str = "prior_test"
    out_str = "multinest/" + save_str + "/" + save_str
    json_str = "multinest/" + save_str + "/params.json"

    # Run the sampler with CEvNS, BRN, and SS.
    pymultinest.run(loglike, prior_stat, 4,
                    outputfiles_basename=out_str,
                    resume=False, verbose=True, n_live_points=2000, evidence_tolerance=0.1,
                    sampling_efficiency=0.8)

    # Save the parameter names to a JSON file.
    params_stat = ["cevns_norm", "ss_norm", "BRN_prompt_norm", "BRN_delayed_norm"]
    json.dump(params_stat, open(json_str, 'w'))







if __name__ == '__main__':


    RunMultinest()



