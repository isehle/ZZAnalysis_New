import os
import helper_functions

import uproot as up

import matplotlib.pyplot as plt
import numpy as np

import yaml

from tqdm import tqdm


def get_init_hists(sample_dict, hist_props, extra_props, fstates, regions, lumi):
    all_props = hist_props | extra_props
    hists = {}
    for region in tqdm(regions, desc = "Regions", position = 0):
        hists[region] = {}
        all_hist_info = {prop: helper_functions.get_hist_info(cfg, region, prop) for prop in hist_props}
        for proc, proc_path in tqdm(sample_dict.items(), desc = "Processes", position = 1, leave = False):
            df = helper_functions.get_df(proc_path, region)
            df = helper_functions.def_cols(df, region, all_props, proc)

            hists[region][proc] = {}

            for fs_key, fs_val in tqdm(fstates.items(), desc = "Final States", position = 2, leave = False):
                hists[region][proc][fs_key] = {}
                for prop in tqdm(hist_props, desc = "Properties", position = 3, leave = False):
                    hist_info = all_hist_info[prop]
                    hists[region][proc][fs_key][prop] = helper_functions.get_hist(df, prop, proc, hist_info, fs=fs_val, lumi=lumi)

    return hists

def polHandler(init_hists, fs, final_hists):
    for weight in ["nominal", "up", "down"]:
        ZLZL = init_hists["ZLZL"][fs][weight]
        ZLZT = init_hists["ZLZT"][fs][weight]
        ZTZT = init_hists["ZTZT"][fs][weight]

        ZLZL.Scale(1/ZLZL.Integral())
        ZLZT.Scale(1/ZLZT.Integral())
        ZTZT.Scale(1/ZTZT.Integral())

        hist_name = lambda name: name if weight=="nominal" else name + "_QCDscale_VV" + weight.capitalize()

        ZLZL_qq = ZLZL.Clone(hist_name("ZLZL_qq"))
        ZLZT_qq = ZLZT.Clone(hist_name("ZLZT_qq"))
        ZTZT_qq = ZTZT.Clone(hist_name("ZTZT_qq"))

        ZLZL_gg = ZLZL.Clone(hist_name("ZLZL_gg"))
        ZLZT_gg = ZLZT.Clone(hist_name("ZLZT_gg"))
        ZTZT_gg = ZTZT.Clone(hist_name("ZTZT_gg"))

        final_hists[fs][hist_name("ZLZL_qq")] = ZLZL_qq
        final_hists[fs][hist_name("ZLZT_qq")] = ZLZT_qq
        final_hists[fs][hist_name("ZTZT_qq")] = ZTZT_qq
        final_hists[fs][hist_name("ZLZL_gg")] = ZLZL_gg
        final_hists[fs][hist_name("ZLZT_gg")] = ZLZT_gg
        final_hists[fs][hist_name("ZTZT_gg")] = ZTZT_gg

    return final_hists

def write_hists(hists, outpath, **kwargs):
    outfile = up.recreate(outpath)

    for reg in kwargs["regions"]:
        for prop in kwargs["props"]:
            for fstate in kwargs["fstates"]:
                for proc in kwargs["procs"]:
                    outfile["{}/{}/{}/{}".format(reg, prop, fstate, proc)] = hists[reg][proc][fstate][prop].GetValue()

def main(cfg, args):
    regions = cfg["regions"]
    samples = helper_functions.get_samples(cfg, year=args["year"], era=args["era"])
    lumi    = float(cfg["datasets"]["year_"+str(args["year"])][args["era"]]["Lumi"])

    hist_props  = cfg["hist_info"].keys()
    extra_props = cfg["extra_props"]

    fstates = cfg["fstates"]
    
    init_hists = get_init_hists(samples, hist_props, extra_props, fstates, regions, lumi)

    outpath = os.path.join(cfg["output"]["base_dir"], "rootFiles", str(args["year"]), "allHists_{}_{}{}.root".format(args["year"], args["era"], args["tag"]))

    write_hists(init_hists, outpath=outpath, regions=regions, props=hist_props, procs=samples.keys(), fstates=fstates.keys())

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="")
    parser.add_argument("--reg", choices=("SR", "OS_NoSIP_HighMass", "OS_NoSIP_MidMass", "OS_NoSIP_LowMass", "SS_NoSIP_HighMass", "SS_NoSIP_MidMass", "SS_NoSIP_LowMass"), default="SR")
    parser.add_argument("--prop", default="mass")
    parser.add_argument("--year", choices=(2022, 2023), default=2022, type=int)
    parser.add_argument("--era", choices=("C", "D", "CD", "EFG", "B"), default="EFG")
    parser.add_argument("--tag", default="")
    args = vars(parser.parse_args())

    with open("/afs/cern.ch/user/i/iehle/polZZTo4l_New/CMSSW_13_0_16/src/ZZAnalysis/NanoAnalysis/scripts/hist_config.yaml") as config:
        cfg = yaml.safe_load(config)

    main(cfg, args)