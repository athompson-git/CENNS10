from numpy import *
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
normSS = truncnorm(-1.0,1.0,scale=ss_error) #norm(scale=0.008)
normPromptBRN = truncnorm(-1.0,1.0,scale=0.3) # norm(scale=0.3)
normDelayedBRN = truncnorm(-1.0,1.0,scale=1.0) # norm(scale=1.0)

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

# Print out totals.
print("TOTALS:")
print("N_obs = ", sum(obs[cut_crit]))
print("N_ss = ", sum(ss[cut_crit]))
print("N_brn =", sum(brn_prompt[cut_crit] + brn_delayed[cut_crit]))
print("N_cevns = ", sum(cevns[cut_crit])) 

sigmaBrnEPlus = brnpdf_p1sigEnergy - brnpdf_data[:,3]
sigmaBrnEMinus = brnpdf_m1sigEnergy - brnpdf_data[:,3]
skewsBrnE = abs(sigmaBrnEPlus) - abs(sigmaBrnEMinus)
signBrnE = sign(skewsBrnE)
snBrnE = skewnorm(skewsBrnE)

sigmaBrnTPlus = brnpdf_p1sigTiming - brnpdf_data[:,3]
sigmaBrnTMinus = brnpdf_m1sigTiming - brnpdf_data[:,3]
skewsBrnT = abs(sigmaBrnTPlus) - abs(sigmaBrnTMinus)
signBrnT = sign(skewsBrnT)
snBrnT = skewnorm(skewsBrnT)

sigmaCEvNSF90Plus = cevnspdf_p1sigF90 - cevnspdf_data[:,3]
sigmaCEvNSF90Minus = cevnspdf_m1sigF90 - cevnspdf_data[:,3]
skewCEvNSF90 = abs(sigmaCEvNSF90Plus) - abs(sigmaCEvNSF90Minus)
signCEvNSF90 = sign(skewCEvNSF90)
snCEvNSF90 = skewnorm(skewCEvNSF90)

sigmaCEvNSTiming = cevnspdfCEvNSTiming - cevnspdf_data[:,3]
signCEvNSTiming = sign(sigmaCEvNSTiming)
normCEvNSTiming = norm(scale=abs(sigmaCEvNSTiming))

sigmaBRNTWidth = brnpdfBRNTimingWidth - brnpdf_data[:,3]
signBRNTWidth = sign(sigmaBRNTWidth)
normBRNTWidth = norm(scale=abs(sigmaBRNTWidth))



def prior_null(cube, n, d):
    cube[0] = cube[0]  # SS norm
    cube[1] = 0.3*sqrt(2)*erfinv(2*cube[1]-1)  # BRN prompt norm
    cube[2] = sqrt(2)*erfinv(2*cube[2]-1)  # BRN delayed norm
    cube[3] = cube[3]  # BRN E dist
    cube[4] = cube[4]  # BRN ttrig mean
    cube[5] = 0.5*(cube[5]+1)  # BRN ttrig width

def prior(cube, n, d):
    cube[0] = 2*cube[0]  # cevns normalization
    cube[1] = cube[1]  # SS norm
    cube[2] = 0.3*sqrt(2)*erfinv(2*cube[2]-1)  # BRN prompt norm
    cube[3] = sqrt(2)*erfinv(2*cube[3]-1)  # BRN delayed norm
    cube[4] = cube[4]  # CEvNS F90 E Dependence
    cube[5] = 0.5*(cube[5]+1)  # CEvNS ttrig mean
    cube[6] = cube[6]  # BRN E dist
    cube[7] = cube[7]  # BRN ttrig mean
    cube[8] = 0.5*(cube[8]+1)  # BRN ttrig width

# Adjust BRN and CEvNS PDFs with systematics
def events_gen(cube):
    # Systematically adjust BRN PDF
    brn_syst = brn_prompt + sigmaBrnEPlus*nan_to_num(snBrnE.ppf(cube[6])) \
                        + sigmaBrnTPlus*nan_to_num(snBrnT.ppf(cube[7])) \
                        + signBRNTWidth*nan_to_num(normBRNTWidth.ppf(cube[8]))

    # Systematically adjust BRN norm
    brn_syst = (1+cube[2])*brn_syst + (1+cube[3])*brn_delayed


    # Systematically adjust CEvNS PDF
    cevns_syst = cevns + sigmaCEvNSF90Plus*nan_to_num(snCEvNSF90.ppf(cube[4])) \
                       + signCEvNSTiming*nan_to_num(normCEvNSTiming.ppf(cube[5]))

    # systematically adjust CEvNS norm
    cevns_syst = cube[0]*cevns_syst

    # Systematically adjust SS norm
    ss_syst = ss + nan_to_num(normSS.ppf(cube[1]))

    return brn_syst + cevns_syst + ss_syst

# No CEvNS
def events_gen_null(cube):
    # Systematically adjust BRN PDF


    brn_syst = brn_prompt + nan_to_num(sigmaBrnEPlus*snBrnE.ppf(cube[3])) \
                        + nan_to_num(sigmaBrnTPlus*snBrnT.ppf(cube[4])) \
                        + signBRNTWidth*nan_to_num(normBRNTWidth.ppf(cube[5]))

    # Systematically adjust BRN norm
    brn_syst = (1+cube[1])*brn_syst + (1+cube[2])*brn_delayed

    # Systematically adjust SS norm
    ss_syst = ss + nan_to_num(normSS.ppf(cube[0]))

    return brn_syst + ss_syst

def prior_stat(cube, n, d):
    cube[0] = 2*cube[0] #1+0.13*sqrt(2)*erfinv(2*cube[0]-1)  # cevns normalization
    cube[1] = normSS.ppf(cube[1])  # SS norm
    cube[2] = normPromptBRN.ppf(cube[2])  # BRN prompt norm
    cube[3] = normDelayedBRN.ppf(cube[3])  # BRN delayed norm

def prior_stat_null(cube, n, d):
    cube[0] = normSS.ppf(cube[0])  # SS norm
    cube[1] = normPromptBRN.ppf(cube[1])  # BRN prompt norm
    cube[2] = normDelayedBRN.ppf(cube[2])  # BRN delayed norm 

# Adjust CEvNS and background PDFs (stats only)
def events_gen_stat(cube):
    # Systematically adjust BRN norm
    brn_syst = (1+cube[2])*brn_prompt + (1+cube[3])*brn_delayed

    # systematically adjust CEvNS norm
    cevns_syst = cube[0]*cevns

    # Systematically adjust SS norm
    ss_syst = (1+cube[1])*ss #+ nan_to_num(normSS.ppf(cube[1])) #(1+cube[1])*ss

    #print("Events gen stat:")
    #print("CEvNS = ", sum(cevns_syst))
    #print("BRN (Prompt) = ", sum((1+cube[2])*brn_prompt))
    #print("BRN (Delayed) = ", sum((1+cube[3])*brn_delayed))
    #print("ss = ", sum(ss_syst))
    return (brn_syst + cevns_syst + ss_syst).clip(min=0.0)

# No CEvNS
def events_gen_stat_null(cube):
    # Systematically adjust BRN norm
    brn_syst = (1+cube[1])*brn_prompt + (1+cube[2])*brn_delayed

    # Systematically adjust SS norm
    ss_syst = (1+cube[0])*ss #+ nan_to_num(normSS.ppf(cube[0])) #(1+cube[0])*ss
    #print("Events gen stat null:")
    #print("BRN = ", sum(brn_syst[(timing < 1.5) & (f90 < 0.7) & (keVee > 10) & (keVee < 35)]))
    #print("ss = ", sum(ss_syst[(timing < 1.5) & (f90 < 0.7) & (keVee > 10) & (keVee < 35)]))

    return (brn_syst + ss_syst).clip(min=0.0)



def poisson(obs, theory):
    ll = 0.
    for i in range(entries):
        if cut_crit[i]:
            ll += obs[i] * log(theory[i]) - theory[i] - gammaln(obs[i]+1)
    return ll


def PrintSignificance():
    print("N_obs = ", sum(obs))

    # F90 < 0.7 + timing, energy cuts
    bf_cuts = [0.622892833389216638,
               0.368517782171544350,
               0.868492585838280373,
               0.297726108731910744]
    bf_cuts_null = [0.304488,
                    1.01207,
                    -1.30858]
    # timing cut, 10 < E < 30
    bf_cuts = [0.838311625065430666,
               0.490974704559929531,
               0.377003040375978038,
               -0.171995834624762561]
    bf_cuts_null = [0.488577827140682708,
                    0.510681793249259330,
                    0.618524854074566810]
    # timing cut, 10 < E < 40
    bf_cuts = [0.194894161460010706,
               0.335561312352763375,
               0.220053713939765444,
              -0.711690709083789508]
    bf_cuts_null = [0.349957301896813067,
                    0.438351803312445942,
                    0.116406455619277163]
    # timing cut, 20 < E < 40
    bf_cuts = [0.680395611692124591,
               0.324972935971696186,
               0.224122897538308663,
               0.185560868316156458]
    bf_cuts_null = [0.325523933493212070,
                    0.263680611878715387,
                   -0.599647883870568665]
    bf = [0.123846624002471128E+01,
         0.452881324148829534E+00,
        -0.456548168897477208E-01,
        -0.769410027256724915E+00,
         0.772841534230718108E+00,
         0.728183973645965210E+00,
         0.231557953329980415E+00,
         0.391704391708963318E+00,
         0.889732421103285764E+00]

    bf_null = [0.482013145961564693E+00,
               0.121184995546275806E+00,
               -0.442341684265830148E+00,
               0.899627672102022852E-01,
               0.361797744959265311E+00,
               0.854267134600862010E+00]
    # Truncated gaussian
    bf_stat = [0.128203949389575733E+01,
              -0.757751720547599188E-02,
               0.928830540200280969E-01,
              -0.681121212215910043E+00]
    bf_stat_null = [-0.799580130637969101E-02,
                     0.253213583049654078E+00,
                    -0.514351228113789194E+00]
    # Unconstrained Gaussian
    bf_stat = [0.168960153287222759E+01,
              -0.312937517992761469E-01,
               0.780942325684447630E-01,
              -0.970385882467374672E+00]
    bf_stat_null = [-0.147905160635425168E-01,
                     0.245574237324468231E+00,
                    -0.460897530294036739E+00]
    # Get ratio test
    print("Default Significance (stat+syst):")
    syst_stat_q = sqrt(abs(2*(-poisson(obs, brn_delayed+brn_prompt+ss+cevns) \
                             + poisson(obs, brn_delayed+brn_prompt+ss))))
    print(syst_stat_q)
    print("Significance (stat+syst):")
    syst_stat_q = sqrt(abs(2*(-poisson(obs, events_gen(bf)) \
                             + poisson(obs, events_gen_null(bf_null)))))
    print(syst_stat_q)

    print("Significance (stat):")
    stat_q = sqrt(abs(2*(-poisson(obs, events_gen_stat(bf_stat)) \
                        + poisson(obs, events_gen_stat_null(bf_stat_null)))))
    print(stat_q)

    print("Significance (stat cuts):")
    stat_q_cuts = sqrt(abs(2*(-poisson(obs, events_gen_stat(bf_cuts)) \
                        + poisson(obs, events_gen_stat_null(bf_cuts_null)))))
    print(stat_q_cuts)




if __name__ == '__main__':

    # Define the log-likelihood for MultiNest.
    def loglike(cube, ndim, nparams):
        n_signal = events_gen_stat(cube)
        ll = 0.0
        for i in range(entries):
            if cut_crit[i]:
                ll += obs[i] * log(n_signal[i]) - n_signal[i] - gammaln(obs[i]+1)
        return sum(ll)

    save_str = "cenns10_stat_asimov_truncGauss"
    out_str = "multinest/" + save_str + "/" + save_str
    json_str = "multinest/" + save_str + "/params.json"

    # Run the sampler.
    pymultinest.run(loglike, prior_stat, 4,
                    outputfiles_basename=out_str,
                    resume=False, verbose=True, n_live_points=1000, evidence_tolerance=0.5,
                    sampling_efficiency=0.8)

    # Save the parameter names to a JSON file.
    params = ["cevns_norm", "ss_norm", "BRN_prompt_norm", "BRN_delayed_norm", "CEvNS_F90_E",
              "CEvNS_t_mean", "BRN_E", "BRN_t_mean", "BRN_t_width"]
    params_null = ["ss_norm", "BRN_prompt_norm", "BRN_delayed_norm",
                   "BRN_E", "BRN_t_mean", "BRN_t_width"]
    params_stat = ["cevns_norm", "ss_norm", "BRN_prompt_norm", "BRN_delayed_norm"]
    params_stat_null = ["ss_norm", "BRN_prompt_norm", "BRN_delayed_norm"]
    json.dump(params_stat, open(json_str, 'w'))



