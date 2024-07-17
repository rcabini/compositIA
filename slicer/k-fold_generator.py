import os, sys
import numpy as np
from sklearn.model_selection import KFold
from glob import glob

#---------------------------------------------------------------------------

def main():
    names = [os.path.basename(f) for f in glob('/home/debian/compositIA/DataNIFTI/Images/*')]
    f = open('k-fold-test.txt', 'w')
    K_FOLDS = 5
    kf = KFold(n_splits=K_FOLDS, random_state=33, shuffle=True)
    
    for k, (train_index, test_index) in enumerate(kf.split(names)):
        TEST_NAME = [names[ff] for ff in test_index]
        print(*TEST_NAME, file = f)


if __name__=="__main__":
    main()

