import pickle
from sklearn import metrics

import os

import PyIO
import PyPluMA
class AUCPlugin:
    def input(self, infile):
       self.parameters = PyIO.readParameters(infile)
    def run(self):
        pass
    def output(self, out_file):
      #inputfile = open("torches.pkl", 'rb')
      inputfile = open(PyPluMA.prefix()+"/"+self.parameters["torches"], 'rb')
      #GRID_DIR = 'data/masif_test/prepare_energies_16R/07-grid/'
      if ("grid" in self.parameters):
       GRID_DIR = PyPluMA.prefix()+"/"+self.parameters["grid"]#'data/masif_test/prepare_energies_16R/07-grid/'
       ppi_list = os.listdir(GRID_DIR)
       ppi_list = [x.split('.npy')[0] for x in ppi_list if 'resnames' not in x and '.ref' not in x]

       labels = [0 if 'neg' in x else 1 for x in ppi_list]
      else:
       infile2 = open(PyPluMA.prefix()+"/"+self.parameters["attn"], 'rb')
       df = pickle.load(infile2)
       labels = list(df['label'])

      output = pickle.load(inputfile)
      pred_probabilities = output.cpu().detach().numpy()
      auc = metrics.roc_auc_score(labels, pred_probabilities)
      print(f"AUC: {1-auc}")

      ## Save the file with all scores
      if ("grid" in self.parameters):
       all_ppis = ppi_list
       #out_file = "PIsToN_scores.csv"
       with open(out_file, 'w') as out:
        out.write("PPI,score,label\n")
        for i in range(len(pred_probabilities)):
           out.write(f"{all_ppis[i]},{pred_probabilities[i]},{labels[i]}\n")

