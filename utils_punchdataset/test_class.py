from utils_punchdataset import loader
import os
from tqdm import tqdm


path = os.path.join("D:\\Arbeit\\V32\\punch_dataset","LookUp.csv")
test=loader.punch_data(path)
train,test=test.split_contiunius(0.1)

for i in tqdm(range(12901)):
    test.loadforce(i)