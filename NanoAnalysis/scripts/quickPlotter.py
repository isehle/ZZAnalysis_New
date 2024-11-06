import matplotlib.pyplot as plt
import mplhep as hep

import uproot as up
import numpy as np

import yaml

import os
import datetime
from pathlib import Path

import helper_functions

def plotter(cfg, args, prop, reg, hists, counts, errors, outfile, **kwargs):    
    prop_info = cfg["hist_info"][prop]

    if prop == "mass":
        mass_reg = reg.split("_")[-1] if reg != "SR" else "SR"
        if mass_reg == "HighMass": mass_reg = "SR"
        ylabel = helper_functions.to_raw_string(prop_info[mass_reg]["ylabel"])

        prop_info = cfg["hist_info"][prop][mass_reg]
    else:
        ylabel = helper_functions.to_raw_string(prop_info["ylabel"])


    fs = kwargs["fstate"]
    mc_counts = counts[fs]["MC"]
    mc_errors = errors[fs]["MC"]

    pol_counts = counts[fs]["Pol"]
    pol_errors = errors[fs]["Pol"]

    pol_labels = [
        r"$q\bar{q} \rightarrow Z_L Z_L$",
        r"$q\bar{q} \rightarrow Z_L Z_T$",
        r"$q\bar{q} \rightarrow Z_T Z_T$",
        r"$(q\bar{q} \rightarrow ZZ)_{LO}$"
    ]
    new_pol_labels = []
    for lab, count, err in zip(pol_labels, pol_counts, pol_errors):
        new_lab = lab + ": " + str(pol_counts[count]) + r" $\pm$ " + str(round(pol_errors[err].sum(), 2))
        new_pol_labels.append(new_lab)

    total_mc_counts = hists[fs]["MC"]["ZZ_NLO"][0]+hists[fs]["MC"]["ggZZ"][0]+hists[fs]["MC"]['DY'][0]+hists[fs]["MC"]["TT"][0]+hists[fs]["MC"]["WZ"][0]+hists[fs]["MC"]['H'][0]+hists[fs]["MC"]["VVV"][0]
    total_mc_errors = np.sqrt(np.sum([err**2 for err in mc_errors.values()], axis=0))

    bin_edges = hists[fs]["MC"]['ZZ_NLO'][1]
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    labels=[
            r"$gg\rightarrow ZZ$",
            r"$(q\bar{q}\rightarrow ZZ)_{NLO}$",
            r"$DY$",
            r"$t\bar{t}$",
            r"$WZ$",
            r"$H$",
            r"$VVV$"
        ]
    new_labels = []
    for lab, count, err in zip(labels, mc_counts, mc_errors):
        new_lab = lab + ": " + str(mc_counts[count]) + r" $\pm$ " + str(round(mc_errors[err].sum(),2))
        new_labels.append(new_lab)

    hep.style.use("CMS")
    ratio_fig_style = cfg["plot_styling"]["ratio_fig_style"]

    if not prop_info["Blind"]:
        fig, (ax, rax) = plt.subplots(2, 1, sharex=True, **ratio_fig_style)
        fig.subplots_adjust(hspace=0.07)
    else:
        fig, ax = plt.subplots()

    hep.histplot(
        list(hists[fs]["MC"].values()),
        stack=True,
        histtype='fill',
        label = new_labels,
        ax = ax,
        color = kwargs["fill_colors"],
        edgecolor = kwargs["line_colors"]
    )

    hep.histplot(
        list(hists[fs]["Pol"].values()),
        histtype="step",
        label = new_pol_labels,
        ax = ax,
        color = ["magenta", "lime", "red", "black"]
    )

    hatch_style = cfg["plot_styling"]["hatch_style"]
    ax.fill_between(x=bin_centers, y1 = total_mc_counts - total_mc_errors, y2 = total_mc_counts + total_mc_errors, label = "Stat. Unc.", step='mid', **hatch_style)

    if args["lumi_tag"] == 0:
        lumi_tag = cfg["datasets"]["year_"+str(args["year"])][args["era"]]["Lumi"]
        lumi_tag = round(float(lumi_tag)*1e-3)
    else:
        lumi_tag = args["lumi_tag"]

    if args["year"] != -1:
        hep.cms.label(label="Work in Progress", year=args["year"], lumi = lumi_tag, com=13.6, data=True, ax=ax)
    else:
        hep.cms.label(label="Work in Progress", lumi = lumi_tag, com=13.6, data=True, ax=ax)

    if not prop_info["Blind"]:
        # Data uncert
        ax.errorbar(bin_centers,
                    hists[fs]["Data"][0],
                    yerr = np.sqrt(hists[fs]["Data"][0]),
                    color = "black",
                    fmt = "o",
                    label = r"Data: {} $\pm$ {}".format(counts[fs]["Data"], round(np.sqrt(counts[fs]["Data"]),2)),
                    markersize = 3)

    xlabel = helper_functions.to_raw_string(prop_info["xlabel"])

    ax.set_ylabel(ylabel)

    ax.legend(ncol=2, fontsize="x-small")

    if not prop_info["Blind"]:
        errorbar_style = cfg["plot_styling"]["errorbar_style"]
        rax.fill_between(x=bin_centers, y1= 1 - total_mc_errors/total_mc_counts, y2 = 1 + total_mc_errors/total_mc_counts, step='mid', **hatch_style)
        rax.errorbar(x=bin_centers, y=hists[fs]["Data"][0]/total_mc_counts, yerr=np.sqrt(hists[fs]["Data"][0])/total_mc_counts, **errorbar_style)

        rax.set_ylim(0, 2)
        rax.set_ylabel('Data / MC')
        rax.set_xlabel(xlabel)
        rax.autoscale(axis='x', tight=True)
    else:
        ax.set_xlabel(xlabel)

    hep.rescale_to_axessize(ax, 10, 10/1.62)

    fig.savefig(outfile)

def main(cfg, args):
    regions = cfg["regions"]
    props = cfg["hist_info"].keys()
    procs = cfg["datasets"]["MC_Procs"] | {"Data": "Data"} | {"ZLZL": "ZLZL", "ZLZT": "ZLZT", "ZTZT": "ZTZT", "ZZ_LO": "ZZ_LO"}
    fstates = cfg["fstates"].keys()

    fill_colors = [cfg["plot_styling"]["mc_colors"][proc]["fill"] for proc in cfg["datasets"]["MC_Procs"].keys()]
    line_colors = [cfg["plot_styling"]["mc_colors"][proc]["line"] for proc in cfg["datasets"]["MC_Procs"].keys()]

    if args["infile"] == "":
        infile = os.path.join(cfg["output"]["base_dir"], "rootFiles", str(args["year"]), "allHists_{}_{}{}.root".format(args["year"], args["era"], args["tag"]))
    else:
        infile = args["infile"]

    all_hists, all_counts, all_errors = helper_functions.read_hists_and_counts(infile, regions=regions, props=props, procs=procs, fstates=fstates)

    for reg in regions:
        for prop in props:
            for fstate in fstates:
                hists  = all_hists[reg][prop]
                counts = all_counts[reg][prop]
                errors = all_errors[reg][prop]

                #outdir = os.path.join(cfg["output"]["base_dir"], "plots", str(args["year"]), args["era"], reg)
                if args["year"] != -1:
                    outdir = os.path.join(cfg["output"]["plot_dir"], str(args["year"]), args["era"], reg, fstate)
                else:
                    outdir = os.path.join(cfg["output"]["plot_dir"], "Full", reg, fstate)
                Path(outdir).mkdir(parents=True, exist_ok=True)
                outfile = os.path.join(outdir, prop+args["tag"]+".png")

                plotter(cfg, args, prop, reg, hists, counts, errors, outfile, fill_colors = fill_colors, line_colors = line_colors, fstate=fstate)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="")
    parser.add_argument("--reg", choices=("SR", "OS_NoSIP_HighMass", "OS_NoSIP_MidMass", "OS_NoSIP_LowMass", "SS_NoSIP_HighMass", "SS_NoSIP_MidMass", "SS_NoSIP_LowMass"), default="SR")
    parser.add_argument("--prop", default="mass")
    parser.add_argument("--year", choices=(2022, 2023, -1), default=2022, type=int)
    parser.add_argument("--era", choices=("C", "D", "CD", "EFG", "B", "CD", "Full"), default="EFG")
    parser.add_argument("--tag", default="")
    parser.add_argument("--infile", default="")
    parser.add_argument("--lumi_tag", default=0.)
    args = vars(parser.parse_args())

    with open("/afs/cern.ch/user/i/iehle/polZZTo4l_New/CMSSW_13_0_16/src/ZZAnalysis/NanoAnalysis/scripts/hist_config.yaml") as config:
        cfg = yaml.safe_load(config)

    main(cfg, args)
