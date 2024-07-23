import os, sys
import numpy as np
from sklearn.model_selection import KFold
from glob import glob
import argparse

#---------------------------------------------------------------------------

def main(args):
    names = [os.path.basename(f) for f in glob(os.path.join(args.data_folder, "Images/*"))]
    f = open(os.path.join(args.data_folder, 'k-fold-test.txt'), 'w')
    K_FOLDS = args.k
    kf = KFold(n_splits=K_FOLDS, random_state=33, shuffle=True)
    
    for k, (train_index, test_index) in enumerate(kf.split(names)):
        TEST_NAME = [names[ff] for ff in test_index]
        print(*TEST_NAME, file = f)


if __name__=="__main__":
    """Read command line arguments"""
	parser = argparse.ArgumentParser()
	parser.add_argument("data_folder", help='Path to dataset')
	parser.add_argument("k", help='Numer of k-folds')
	args = parser.parse_args()
	main(args)

