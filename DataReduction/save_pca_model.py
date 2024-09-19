import os
import numpy
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from fnmatch import fnmatch
from tqdm import tqdm
from natsort import index_natsorted
import matplotlib
#import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import pickle as pk


if __name__ == '__main__':
    acc=np.load("/media/maxi/T7 Shield/Arbeit/DataUtils/PCA/a_vo_filtered.npz")
    ae=np.load("/media/maxi/T7 Shield/Arbeit/DataUtils/PCA/ae_vo_filtered.npz")
    f=np.load("/media/maxi/T7 Shield/Arbeit/DataUtils/PCA/force_vo_filtered.npz")
    im = np.load("/media/maxi/T7 Shield/Arbeit/DataUtils/PCA/im_vo_filtered.npz")

    # Standardizing the features
    acc = normalize(acc['acc'])
    ae = normalize(ae['ae'])
    f = normalize(np.nan_to_num(f['f']))
    im = im['im']/255.


    # #TEST PCA
    pca_test_im = PCA(n_components=2500)  #
    pca_test_im.fit(np.nan_to_num(im))
    pca_sum_im = np.cumsum(pca_test_im.explained_variance_ratio_)
    index_99_im = pca_sum_im < 0.90
    val_im, counts_im = np.unique(index_99_im,return_counts=True)
    print (np.unique(index_99_im,return_counts=True))

    im_df = pd.DataFrame(data=np.nan_to_num(im))
    trans_im = pca_test_im.fit_transform(im_df)
    back = pca_test_im.inverse_transform(trans_im)

    plt.plot(pca_sum_im)
    plt.xlabel("Number of components")
    plt.ylabel("Cum variance")
    plt.title('PCA_IM')
    plt.vlines(counts_im[1],0,1.2,color='red')
    plt.text(x=counts_im[1]+2,y=1.15,s='AE_IM='+str(counts_im[1]))
    plt.show()
    #plt.savefig("pca_ae.png")
    pca_im = PCA(n_components=counts_im[1])
    pk.dump(pca_im, open("/media/maxi/T7 Shield/Arbeit/DataUtils/PCA/pca_im_vo.pkl", "wb"))


    # #TEST PCA
    pca_test_ae = PCA(n_components=2500)  #
    pca_test_ae.fit(np.nan_to_num(ae))
    pca_sum_ae = np.cumsum(pca_test_ae.explained_variance_ratio_)
    index_99_ae = pca_sum_ae < 0.99
    val_ae, counts_ae = np.unique(index_99_ae,return_counts=True)
    print (np.unique(index_99_ae,return_counts=True))

    # plt.plot(pca_sum_ae)
    # plt.xlabel("Number of components")
    # plt.ylabel("Cum variance")
    # plt.title('PCA_AE')
    # plt.vlines(counts_ae[1],0,1.2,color='red')
    # plt.text(x=counts_ae[1]+2,y=1.15,s='AE_PCA='+str(counts_ae[1]))
    #plt.show()
    #plt.savefig("pca_ae.png")

    pca_ae = PCA(n_components=counts_ae[1])
    pk.dump(pca_ae, open("/media/maxi/T7 Shield/Arbeit/DataUtils/PCA/pca_ae_vo.pkl", "wb"))

    # #TEST PCA
    pca_test_f = PCA(n_components=500)  #
    pca_test_f.fit(np.nan_to_num(f))
    pca_sum_f = np.cumsum(pca_test_f.explained_variance_ratio_)
    index_99_f = pca_sum_f < 0.99
    val_f, counts_f = np.unique(index_99_f,return_counts=True)
    print (np.unique(index_99_f,return_counts=True))
    #
    # #pca = PCA(n_components=counts[1])
    #

    # plt.plot(pca_sum_f)
    # plt.xlabel("Number of components")
    # plt.ylabel("Cum variance")
    # plt.title('PCA_Force')
    # plt.vlines(counts_f[1],0,1.2,color='red')
    # plt.text(x=counts_f[1] + 520, y=0.75, s='F_PCA='+str(counts_f[1]))
    #plt.show()
    #plt.savefig("pca_f.png")

    pca_f = PCA(n_components=counts_f[1])
    pk.dump(pca_f, open("/media/maxi/T7 Shield/Arbeit/DataUtils/PCA/pca_f_vo.pkl", "wb"))

    # #TEST PCA
    pca_test_a = PCA(n_components=500)  #
    pca_test_a.fit(np.nan_to_num(acc))
    pca_sum_a = np.cumsum(pca_test_a.explained_variance_ratio_)
    index_99_a = pca_sum_a < 0.99
    val_a, counts_a = np.unique(index_99_a,return_counts=True)
    print (np.unique(index_99_a,return_counts=True))
    #
    # #pca = PCA(n_components=counts[1])
    #

    # plt.plot(pca_sum_a)
    # plt.xlabel("Number of components")
    # plt.ylabel("Cum variance")
    # plt.title('PCA_Acc')
    # #plt.vlines(counts_a[1],0,1.2,color='red')
    # plt.text(x=counts_a[1] + 1050, y=1.15, s='Acc_PCA='+str(counts_a[1]))
    # #plt.show()
    # plt.savefig("pca_ll.png")

    pca_a = PCA(n_components=counts_a[1])
    pk.dump(pca_a, open("/media/maxi/T7 Shield/Arbeit/DataUtils/PCA/pca_a_vo.pkl", "wb"))
    #
    # force = 21
    # ae = 476
    # a = 118