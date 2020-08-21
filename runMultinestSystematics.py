import numpy as np
from numpy import log10, exp, log, genfromtxt, sqrt, pi, sign, nan_to_num
from scipy.special import erfinv, gammaln
from scipy.stats import skewnorm, norm, truncnorm
import matplotlib.pyplot as plt

import pymultinest
import mpi4py

import json


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

# Define stats CDFs for priors
ss_error = np.sqrt(np.sum(ss)/5)
normSS = norm(scale=ss_error)
normPromptBRN = norm(scale=0.3)
normDelayedBRN = norm(scale=1.0)

# Flat bins
entries = obs_data.shape[0]
keVee = obs_data[:,0]
f90 = obs_data[:,1]
timing = obs_data[:,2]


sigmaBrnEPlus = brnpdf_p1sigEnergy - brnpdf_data[:,3]
sigmaBrnEMinus = brnpdf_m1sigEnergy - brnpdf_data[:,3]
skewsBrnE = nan_to_num((abs(sigmaBrnEPlus) - abs(sigmaBrnEMinus)) / (abs(sigmaBrnEPlus) + abs(sigmaBrnEMinus)))
signBrnE = sign(sigmaBrnEPlus)
sigmaBrnE = abs(brnpdf_p1sigEnergy - brnpdf_m1sigEnergy)
snBrnE = skewnorm(skewsBrnE, scale=sigmaBrnE.clip(min=0.1))

sigmaBrnTPlus = brnpdf_p1sigTiming - brnpdf_data[:,3]
sigmaBrnTMinus = brnpdf_m1sigTiming - brnpdf_data[:,3]
skewsBrnT = nan_to_num((abs(sigmaBrnTPlus) - abs(sigmaBrnTMinus)) / (abs(sigmaBrnTPlus) + abs(sigmaBrnTMinus)))
signBrnT = sign(sigmaBrnTPlus)
sigmaBrnT = abs(brnpdf_p1sigTiming - brnpdf_m1sigTiming)
snBrnT = skewnorm(skewsBrnT, scale=sigmaBrnT.clip(min=0.1))

sigmaCEvNSF90Plus = cevnspdf_p1sigF90 - cevnspdf_data[:,3]
sigmaCEvNSF90Minus = cevnspdf_m1sigF90 - cevnspdf_data[:,3]
skewCEvNSF90 = nan_to_num((abs(sigmaCEvNSF90Plus) - abs(sigmaCEvNSF90Minus)) / (abs(sigmaCEvNSF90Plus) + abs(sigmaCEvNSF90Minus)))
signCEvNSF90 = sign(sigmaCEvNSF90Plus)
sigmaCEvNSF90 = abs(cevnspdf_p1sigF90 - cevnspdf_m1sigF90)
snCEvNSF90 = skewnorm(skewCEvNSF90, scale=sigmaCEvNSF90.clip(min=0.1))

sigmaCEvNSTiming = cevnspdfCEvNSTiming - cevnspdf_data[:,3]
signCEvNSTiming = sign(sigmaCEvNSTiming)
normCEvNSTiming = norm(scale=abs(sigmaCEvNSTiming).clip(min=0.1))

sigmaBRNTWidth = brnpdfBRNTimingWidth - brnpdf_data[:,3]
signBRNTWidth = sign(sigmaBRNTWidth)
normBRNTWidth = norm(scale=abs(sigmaBRNTWidth).clip(min=0.1))



def prior_null(cube, n, d):
    cube[0] = normSS.ppf(cube[0])  # SS norm
    cube[1] = normPromptBRN.ppf(cube[1])  # BRN prompt norm
    cube[2] = normDelayedBRN.ppf(cube[2])  # BRN delayed norm
    cube[3] = cube[3]  # BRN E dist
    cube[4] = cube[4]  # BRN ttrig mean
    cube[5] = 0.5*(cube[5]+1)  # BRN ttrig width

def prior(cube, n, d):
    cube[0] = 2*cube[0]  # CEvNS norm
    cube[1] = normSS.ppf(cube[1])  # SS norm
    cube[2] = normPromptBRN.ppf(cube[2])  # BRN prompt norm
    cube[3] = normDelayedBRN.ppf(cube[3])  # BRN delayed norm
    cube[4] = cube[4]  # CEvNS F90 E Dependence
    cube[5] = 0.5*(cube[5]+1)  # CEvNS ttrig mean
    cube[6] = cube[6]  # BRN E dist
    cube[7] = cube[7]  # BRN ttrig mean
    cube[8] = 0.5*(cube[8]+1)  # BRN ttrig width

# Adjust BRN and CEvNS PDFs with systematics
def events_gen(cube, report_stats=False):
    # Systematically adjust BRN PDF
    brn_syst = brn_prompt + nan_to_num(signBrnE*snBrnE.ppf(cube[6])) \
                          + nan_to_num(signBrnT*snBrnT.ppf(cube[7])) \
                          + nan_to_num(signBRNTWidth*normBRNTWidth.ppf(cube[8]))

    brn_syst = ((1+cube[2])*brn_syst).clip(min=0.01)
    dbrn_syst = ((1+cube[3])*brn_delayed).clip(min=0.01)


    # Systematically adjust CEvNS PDF
    cevns_syst = cevns + nan_to_num(signCEvNSF90*snCEvNSF90.ppf(cube[4])) \
                       + nan_to_num(signCEvNSTiming*normCEvNSTiming.ppf(cube[5]))

    cevns_syst = (cube[0]*cevns_syst).clip(min=0.01)

    # Systematically adjust SS PDF
    ss_syst = ((1+cube[1])*ss).clip(min=0.01)

    if report_stats:
        print("CEvNS = ", sum(cevns_syst))
        print("SS = ", sum(ss_syst))
        print("Prompt BRN = ", sum(brn_syst))
        print("Delayed BRN = ", dbrn_syst)

    return brn_syst + dbrn_syst + cevns_syst + ss_syst

# No CEvNS
def events_gen_null(cube):
    # Systematically adjust BRN PDF
    brn_syst = brn_prompt + nan_to_num(signBrnE*snBrnE.ppf(cube[3])) \
                          + nan_to_num(signBrnT*snBrnT.ppf(cube[4])) \
                          + nan_to_num(signBRNTWidth*normBRNTWidth.ppf(cube[5]))

    # Systematically adjust BRN norm
    brn_syst = ((1+cube[1])*brn_syst).clip(min=0.01) + ((1+cube[2])*brn_delayed).clip(min=0.01)

    # Systematically adjust SS norm
    ss_syst = ((1+cube[0])*ss).clip(min=0.01)

    return brn_syst + ss_syst



def poisson(obs, theory):
    ll = obs * log(theory) - theory - gammaln(obs+1)
    return sum(ll)

def PrintSignificance():
    an = Analyzer(4, "multinest/cenns10_syst/cenns10_syst")
    bf = an.get_best_fit()['parameters']

    an_null = Analyzer(3, "multinest/cenns10_syst_no_cevns/cenns10_syst_no_cevns")
    bf_null = an_null.get_best_fit()['parameters']
    bf = [an.get_stats()['marginals'][0]['median'], an.get_stats()['marginals'][1]['median'],
          an.get_stats()['marginals'][2]['median'], an.get_stats()['marginals'][3]['median']]
    bf_null = [an_null.get_stats()['marginals'][0]['median'], an_null.get_stats()['marginals'][1]['median'],               an_null.get_stats()['marginals'][2]['median']]

    # Get ratio test
    print("Significance (stat):")
    stat_q = sqrt(abs(2*(-poisson(obs, events_gen(bf)) \
                        + poisson(obs, events_gen_null(bf_null)))))
    print(stat_q)

    print("Best-fit norms:")
    events_gen(bf, report_stats=True)



def RunMultinest():
    def loglike(cube, ndim, nparams):
        n_signal = events_gen(cube)
        ll = obs * log(n_signal) - n_signal - gammaln(obs+1)
        return sum(ll)

    save_str = "cenns10_syst"
    out_str = "multinest/" + save_str + "/" + save_str
    json_str = "multinest/" + save_str + "/params.json"

    # Run the sampler with CEvNS, BRN, and SS.
    pymultinest.run(loglike, prior, 9,
                    outputfiles_basename=out_str,
                    resume=True, verbose=True, n_live_points=2000, evidence_tolerance=0.5,
                    sampling_efficiency=0.8)

    # Save the parameter names to a JSON file.
    params = ["cevns_norm", "ss_norm", "BRN_prompt_norm", "BRN_delayed_norm", "CEvNS_F90_E",
              "CEvNS_t_mean", "BRN_E", "BRN_t_mean", "BRN_t_width"]
    json.dump(params, open(json_str, 'w'))




def RunMultinestNull():
    def loglike(cube, ndim, nparams):
        n_signal = events_gen_null(cube)
        ll = obs * log(n_signal) - n_signal - gammaln(obs+1)
        return sum(ll)

    save_str = "cenns10_syst_no_cevns"
    out_str = "multinest/" + save_str + "/" + save_str
    json_str = "multinest/" + save_str + "/params.json"

    # Run the sampler with just BRN, and SS.
    pymultinest.run(loglike, prior_null, 6,
                    outputfiles_basename=out_str,
                    resume=False, verbose=True, n_live_points=2000, evidence_tolerance=0.5,
                    sampling_efficiency=0.8)

    # Save the parameter names to a JSON file.
    params_null = ["ss_norm", "BRN_prompt_norm", "BRN_delayed_norm",
                   "BRN_E", "BRN_t_mean", "BRN_t_width"]
    json.dump(params_null, open(json_str, 'w'))



if __name__ == '__main__':

    print("Running MultiNest with CEvNS, BRN, and SS components...")

    RunMultinest()

    print("Starting next run for only BRN and SS components (5s)...")

    time.sleep(5.0)

    RunMultinestNull()

    PrintSignificance()



