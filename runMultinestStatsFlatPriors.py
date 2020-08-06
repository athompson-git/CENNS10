from numpy import *
from scipy.special import erfinv, gammaln
from scipy.stats import skewnorm, norm, truncnorm
import matplotlib.pyplot as plt

import pymultinest
import mpi4py

import json
import time

# Read in data
bkgpdf_data = genfromtxt("Data/bkgpdf.txt")
brnpdf_data = genfromtxt("Data/brnpdf.txt")
brndelayedpdf_data = genfromtxt("Data/delbrnpdf.txt")
cevnspdf_data = genfromtxt("Data/cevnspdf.txt")
obs_data = genfromtxt("Data/datanobkgsub.txt")

# Read in systematics PDFs
brnpdf_m1sigTiming = genfromtxt("Data/SystErrors/brnpdf-1sigBRNTimingMean.txt")[:,3]
brnpdf_p1sigTiming = genfromtxt("Data/SystErrors/brnpdf+1sigBRNTimingMean.txt")[:,3]
brnpdf_m1sigEnergy = genfromtxt("Data/SystErrors/brnpdf-1sigEnergy.txt")[:,3]
brnpdf_p1sigEnergy = genfromtxt("Data/SystErrors/brnpdf+1sigEnergy.txt")[:,3]
cevnspdf_m1sigF90 = genfromtxt("Data/SystErrors/cevnspdf-1sigF90.txt")[:,3]
cevnspdf_p1sigF90 = genfromtxt("Data/SystErrors/cevnspdf+1sigF90.txt")[:,3]
cevnspdfCEvNSTiming = genfromtxt("Data/SystErrors/cevnspdfCEvNSTimingMeanSyst.txt")[:,3]
brnpdfBRNTimingWidth = genfromtxt("Data/SystErrors/brnpdfBRNTimingWidthSyst.txt")[:,3]

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

# Define a boolean array for any cuts.
timing_cut = timing < 12
keVee_lo_cut = keVee > 0
keVee_hi_cut = keVee < 140
f90_cut = f90 < 1.0

cut_crit = timing_cut*keVee_lo_cut*keVee_hi_cut*f90_cut

# Define stats CDFs for priors
ss_error = sqrt(sum(ss)/5)/sum(ss) # percent error
normSS = norm(scale=ss_error) #truncnorm(-1.0,1.0,scale=ss_error)
normPromptBRN = norm(scale=0.3) #truncnorm(-1.0,1.0,scale=0.3)
normDelayedBRN = norm(scale=1.0) #truncnorm(-1.0,1.0,scale=1.0)




# Define Priors for MultiNest
def prior_stat(cube, n, d):
    cube[0] = 2*cube[0]  # CEvNS norm
    cube[1] = 2*cube[1]  # SS norm
    cube[2] = 2*(cube[2])  # BRN prompt norm
    cube[3] = 2*(cube[3])  # BRN delayed norm

def prior_stat_null(cube, n, d):
    cube[0] = 2*(cube[0])  # SS norm
    cube[1] = 2*(cube[1])  # BRN prompt norm
    cube[2] = 2*(cube[2])  # BRN delayed norm 




# Generate new PDFs with nuisance-controlled norms
def events_gen_stat(cube, report_stats=False):
    brn_syst = cube[2]*brn_prompt + cube[3]*brn_delayed
    cevns_syst = cube[0]*cevns
    ss_syst = cube[1]*ss

    if report_stats:
        print("N_CEvNS = ", sum(cevns_syst))
        print("N_BRN_PRO = ", sum(cube[2]*brn_prompt))
        print("N_BRN_DEL = ", sum(cube[3]*brn_delayed))
        print("N_SS = ", sum(ss_syst))

    return (brn_syst + cevns_syst + ss_syst).clip(min=0.0)

# Generate new PDFs with nuisance-controlled norms (no CEvNS)
def events_gen_stat_null(cube):
    brn_syst = cube[1]*brn_prompt + cube[2]*brn_delayed
    ss_syst = cube[0]*ss

    return (brn_syst + ss_syst).clip(min=0.0)




def poisson(obs, theory):
    ll = 0.
    for i in range(entries):
        if cut_crit[i]:
            ll += obs[i] * log(theory[i]) - theory[i] - gammaln(obs[i]+1)
    return ll

# TODO(AT): parse the MLE parameters directly from MultiNest output files
def PrintSignificance():
    # Print out totals.
    print("TOTALS:")
    print("N_obs = ", sum(obs[cut_crit]))
    print("N_ss = ", sum(ss[cut_crit]))
    print("N_brn =", sum(brn_prompt[cut_crit] + brn_delayed[cut_crit]))
    print("N_cevns = ", sum(cevns[cut_crit]))

    # Save best-fit (MLE) parameters from MultiNest (in <out>stats.dat)
    # Unconstrained Gaussian
    bf_stat = [0.138304677686178401E+01,
               0.999978273618992497E+00,
               0.100284641508612893E+01,
               0.993472985933852137E+00]
    bf_stat_null = [0.100000134302762644E+01,
                    0.100634787141285598E+01,
                    0.996008220044579451E+00]

    # Get ratio test
    print("Significance (stat):")
    stat_q = sqrt(abs(2*(-poisson(obs, events_gen_stat(bf_stat)) \
                        + poisson(obs, events_gen_stat_null(bf_stat_null)))))

    print(stat_q)
    print("Best Fit Norms:")
    events_gen_stat(bf_stat, report_stats=True)




def RunMultinest():
    def loglike(cube, ndim, nparams):
        n_signal = events_gen_stat(cube)
        ll = cut_crit*(obs * log(n_signal) - n_signal - gammaln(obs+1)) \
             - (cube[1] - 1)**2 / (2 * ss_error**2) \
             - (cube[2] - 1)**2 / (2 * 0.3**2) \
             - (cube[3] - 1)**2 / (2)
        return sum(ll)

    save_str = "cenns10_stat_llConstraint"
    out_str = "multinest/" + save_str + "/" + save_str
    json_str = "multinest/" + save_str + "/params.json"

    # Run the sampler with CEvNS, BRN, and SS.
    pymultinest.run(loglike, prior_stat, 4,
                    outputfiles_basename=out_str,
                    resume=False, verbose=True, n_live_points=1000, evidence_tolerance=0.5,
                    sampling_efficiency=0.8)

    # Save the parameter names to a JSON file.
    params_stat = ["cevns_norm", "ss_norm", "BRN_prompt_norm", "BRN_delayed_norm"]
    json.dump(params_stat, open(json_str, 'w'))




def RunMultinestNull():
    def loglike(cube, ndim, nparams):
        n_signal = events_gen_stat_null(cube)
        ll = cut_crit*nan_to_num(obs * log(n_signal) - n_signal - gammaln(obs+1)) \
             - (cube[0] - 1)**2 / (2 * ss_error**2) \
             - (cube[1] - 1)**2 / (2 * 0.3**2) \
             - (cube[2] - 1)**2 / (2)
        return sum(nan_to_num(ll))

    save_str = "cenns10_stat_null_llConstraint"
    out_str = "multinest/" + save_str + "/" + save_str
    json_str = "multinest/" + save_str + "/params.json"

    # Run the sampler with just BRN, and SS.
    pymultinest.run(loglike, prior_stat_null, 3,
                    outputfiles_basename=out_str,
                    resume=False, verbose=True, n_live_points=1000, evidence_tolerance=0.5,
                    sampling_efficiency=0.8)

    # Save the parameter names to a JSON file.
    params_stat_null = ["ss_norm", "BRN_prompt_norm", "BRN_delayed_norm"]
    json.dump(params_stat_null, open(json_str, 'w'))




if __name__ == '__main__':

    print("Running MultiNest with CEvNS, BRN, and SS components...")

    RunMultinest()

    print("Starting next run for only BRN and SS components (5s)...")

    time.sleep(5.0)

    RunMultinestNull()

    #PrintSignificance()

