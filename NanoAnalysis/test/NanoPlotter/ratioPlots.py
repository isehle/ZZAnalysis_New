import ROOT

import awkward as ak
import numpy as np
import uproot as up

import hist
from hist import Hist

from matplotlib import pyplot as plt
import mplhep as hep

import datetime
from pathlib import Path
import os

ROOT.gInterpreter.Declare("""
using ROOT::RVecF;
using ROOT::RVecB;
RVecF applyRegFilt(RVecF &prop, RVecB &regMask){
    RVecF reg_prop;
    for (int i=0; i<prop.size(); i++){
        if (regMask.at(i) == true){
            reg_prop.push_back(prop.at(i));
        }
    }
    return reg_prop;
}
""")

hist_info = dict(
    mass = dict(
        bins  = int((2000-105)/10),
        start = 105,
        stop  = 2000,
        name  = r"$m_{4l}$",
        label = r"$m_{4l}$",        
        flow  = False
    ),
    pt = dict(
        bins  = 500,
        start = 0,
        stop  = 500,
        name  = r"${p_T}^{4l}$",
        label = r"${p_T}^{4l}$",        
        flow  = False
    ),
    eta = dict(
        bins  = int(14/0.1),
        start = -7,
        stop  = 7,
        name  = r"${\eta}_{4l}$",
        label = r"${\eta}_{4l}$",        
        flow  = False
    ),
    cosTheta1 = dict(
        bins = int(2/0.3),
        start = -1,
        stop = 1,
        name = r"$cos(\theta_1)$",
        label= r"$cos(\theta_1)$",        
        flow = False
    ),
    cosTheta3 = dict(
        bins = int(2/0.3),
        start = -1,
        stop = 1,
        name = r"$cos(\theta_3)$",
        label= r"$cos(\theta_3)$",        
        flow = False
    ),
    cosThetaStar = dict(
        bins = int(2/0.3),
        start = -1,
        stop = 1,
        name = r"$cos(\theta*)$",
        label= r"$cos(\theta*)$",        
        flow = False
    ),
    Z1mass = dict(
        bins  = int((105-70)/1),
        start = 70,
        stop  = 105,
        name  = r"${m}_{ll}$",
        label = r"${m}_{ll}$",        
        flow  = False
    ),
    Z2mass = dict(
        bins  = int((105-70)/1),
        start = 70,
        stop  = 105,
        name  = r"${m}_{ll}$",
        label = r"${m}_{ll}$",        
        flow  = False
    ),
    Lepton_pt = dict(
        bins  = 50,
        start = 0,
        stop  = 500,
        name  = r"$p_T^l$",
        label = r"$p_T^l$",        
        flow  = False
    ),
    Lepton_eta = dict(
        bins  = int((14/0.1)),
        start = -7,
        stop  = 7,
        name  = r"${\eta}^l$",
        label = r"${\eta}^l$",        
        flow  = False
    ),
    Lepton_sip3d = dict(
        bins  = 40,
        start = 0,
        stop  = 40,
        name  = r"${SIP_{3D}}^l$",
        label = r"${SIP_{3D}}^l$",        
        flow  = False
    )
)

def get_df(path):
    df = ROOT.RDataFrame("Events", path)
    df = df.Filter("HLT_passZZ4l")

    Runs = ROOT.RDataFrame("Runs", path)
    
    df = df.Define("genEventSumw", str(Runs.Sum("genEventSumw").GetValue()))
    df = df.Define("weight", "overallEventWeight/genEventSumw") # ZZCand_MCWeight*overallEventWeight/genEventSumw

    return df

def def_cols(df, reg_idx, props):
    cand = "ZZCand" if reg_idx == "SR" else "ZLLCand"
    reg_prop = lambda prop: "applyRegFilt({}_{}, {})".format(cand, prop, reg_idx)

    for prop in props:
        if prop == "weight": continue
        df = df.Define(prop, reg_prop(prop))

    return df

def get_arrs(path, props):
    df = get_df(path)
    df = def_cols(df, "SR", props)

    arrs = ak.from_rdataframe(df, columns=props)

    #good_idx = ak.flatten(np.argwhere(ak.num(arrs['mass']) == 1))
    good_idx = ak.flatten(np.argwhere(ak.num(arrs['cosTheta1']) == 1))

    good_arrs = {}
    for prop in arrs.fields:
        if prop=="weight":
            good_arrs[prop] = arrs[prop][good_idx]
        else:
            good_arrs[prop] = ak.flatten(arrs[prop][good_idx])

    return good_arrs

def get_hists(mg_arrs, pwg_arrs, props):
    mg_weight  = mg_arrs["weight"]
    pwg_weight = pwg_arrs["weight"]

    hists = {}
    for prop in props:
        if prop == "weight": continue
        mg_arr  = mg_arrs[prop]
        pwg_arr = pwg_arrs[prop]

        mg_hist  = hist.Hist(hist.axis.Regular(**hist_info[prop]))
        pwg_hist = hist.Hist(hist.axis.Regular(**hist_info[prop]))

        mg_hist.fill(mg_arr, weight=mg_weight)
        pwg_hist.fill(pwg_arr, weight=pwg_weight)

        mg_hist  *= 27.007e3
        pwg_hist *= 27.007e3

        hists[prop] = {"mg": mg_hist,
                       "pwg": pwg_hist}

    return hists

def plot_dir(cat, reg="SR", base_dir="/eos/user/i/iehle/Analysis/pngs/plots/", date=str(datetime.date.today())):
    if "Hists" not in base_dir:
        direc = os.path.join(base_dir, date, reg, cat)
    else:
        direc = os.path.join(base_dir, date)
    Path(direc).mkdir(parents=True, exist_ok=True)
    return direc

def main(mg_path, pwg_path):
    #props   = ["mass", "pt", "eta", "Z1mass", "Z2mass", "cosTheta1", "cosTheta3", "cosThetaStar", "weight"]
    props   = ["cosTheta1", "cosTheta3", "cosThetaStar", "weight"]

    mg_arrs = get_arrs(mg_path, props)
    pwg_arrs = get_arrs(pwg_path, props)

    hists = get_hists(mg_arrs, pwg_arrs, props)

    direc = plot_dir("ZZTo4l")
    for prop in hists:
        

        mg_hist  = hists[prop]["mg"]
        pwg_hist = hists[prop]["pwg"]

        ratio     = mg_hist.counts()/pwg_hist.counts()
        max_ratio = max(np.max(ratio[np.isfinite(ratio)]), 1)

        fig = plt.figure(figsize=(10, 8))

        main_ax_artists, subplot_ax_artists = mg_hist.plot_ratio(
            pwg_hist,
            rp_ylabel           = r"MG/Powheg",
            rp_num_label        = r"$(q\bar{q} \rightarrow ZZ)_{MG}$",
            rp_denom_label      = r"$(q\bar{q} \rightarrow ZZ)_{Powheg}$",
            rp_uncert_draw_type = "line"
        )
        
        plt.setp(subplot_ax_artists[0].axes, ylim=(0.6,1.1*max_ratio))

        plot_path = os.path.join(direc, f"{prop}_MG_Powheg_Ratio_betterYLim.png")
        plt.savefig(plot_path)

'''def main(hfiles):
    fstates = ["fs_{}".format(state) for state in ["4e", "4mu", "2e2mu"]]
    direc = plot_dir("ZZTo4l")
    for prop, path in hfiles.items():
        with up.open(path) as hfile:
            for fs in fstates:

                mg_hist  = hfile[fs+"/ZZ_LO"]
                pwg_hist = hfile[fs+"/ZZ_NLO"]

                ratio     = mg_hist.values()/pwg_hist.values()
                max_ratio = max(np.max(ratio[np.isfinite(ratio)]), 1)

                errors = ratio*np.sqrt((mg_hist.errors()/mg_hist.values())**2 + (pwg_hist.errors()/pwg_hist.values())**2)

                fig, ax = plt.subplots()

                ax.errorbar(np.linspace(-1, 1, 10), ratio, yerr=errors)

                plot_path = os.path.join(direc, f"{prop}_MG_Powheg_Ratio_betterYLim_{fs}.png")
                plt.savefig(plot_path)'''


if __name__=="__main__":
    #qqZZ_MG     = "/afs/cern.ch/user/i/iehle/polZZTo4l/CMSSW_13_0_16/src/ZZAnalysis/pol_skims/ZZ_hadd_Skim.root"
    #qqZZ_Powheg = "/eos/user/i/iehle/Analysis/MC_2022EE_wCosTheta_wNoSIPCR/ZZTo4l_Chunk2/ZZ4lAnalysis.root"

    qqZZ_MG     = "/afs/cern.ch/user/i/iehle/polZZTo4l_New/CMSSW_13_0_16/src/ZZAnalysis/pol_skims/wCosThetaStar/ZZ_hadd_Skim.root"
    qqZZ_Powheg = "/eos/user/i/iehle/Analysis/MC_2022EE_wCosThetaStar/ZZTo4l_Chunk1/ZZ4lAnalysis.root"

    '''hfiles = dict(
        cosTheta1    = "all_cosTheta1_hists_v2.root",
        cosTheta3    = "all_cosTheta3_hists.root",
        cosThetaStar = "all_cosThetaStar_hists.root"
    )
    main(hfiles)'''
    
    main(qqZZ_MG, qqZZ_Powheg)