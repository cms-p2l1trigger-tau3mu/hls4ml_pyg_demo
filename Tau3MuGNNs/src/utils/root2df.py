# Modified from https://github.com/cms-p2l1trigger-tau3mu/Tau3MuGNNs/blob/master/ProcessROOTFiles.py
# The code may only work perfectly on x86_64.

import pandas as pd
import uproot3 as uproot

import ast
import configparser
from pathlib import Path


class Root2Df(object):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.config_file = data_dir / 'processing.cfg'
        self.set_config()

    def set_config(self):
        print(f"[INFO] Reading configuration file: {self.config_file}")
        cfgparser = configparser.ConfigParser()
        cfgparser.read(self.config_file)

        self.signalsamples = ast.literal_eval(cfgparser.get("general", "signalsamples"))
        self.backgroundsamples = ast.literal_eval(cfgparser.get("general", "backgroundsamples"))
        self.signalvariables = ast.literal_eval(cfgparser.get("filter", "signalvariables"))
        self.backgroundvariables = ast.literal_eval(cfgparser.get("filter", "backgroundvariables"))

    def process_root_file(self, samplename, variables, max_events):
        print("[INFO] Transforming ROOT files into pickle files")
        # open dataset
        print(f"    ... Opening file in input directory using uproot: {samplename}")
        events = uproot.open(samplename)['Ntuplizer/MuonTrackTree']

        # transform file into a pandas dataframe
        print("    ... Processing file using pandas")
        unfiltered_events_df = events.pandas.df(variables, entrystop=max_events, flatten=False)

        out_file = samplename.parent / (samplename.stem + '.pkl')
        print(f"    ... Saving file in output directory: {out_file}")
        unfiltered_events_df.to_pickle(out_file)

    def process_all_samples(self, pos_max, neg_max):
        for sample in self.backgroundsamples:
            self.process_root_file(self.data_dir / sample, self.backgroundvariables, neg_max)

        for sample in self.signalsamples:
            self.process_root_file(self.data_dir / sample, self.signalvariables, pos_max)

    def read_df(self, setting):
        if 'mix' not in setting:
            self.signalsamples.remove('DsTau3muPU0_MTD.root')

        res = {}
        for sample in (self.backgroundsamples + self.signalsamples):
            sample = self.data_dir / sample
            try:
                res[sample.stem] = pd.read_pickle(sample.parent / (sample.stem + '.pkl'))
            except FileNotFoundError:
                print(f'[WARNING] {sample} not found!')
            else:
                print(f'[INFO] {sample} loaded!')
        return res


def main():
    Root2Df(data_dir=Path('../../data/raw')).process_all_samples(pos_max=None, neg_max=None)


if __name__ == '__main__':
    main()
