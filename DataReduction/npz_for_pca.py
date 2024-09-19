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
import cv2
from sklearn.decomposition import PCA



if __name__ == '__main__':
    #matplotlib.use("GTK3Agg")
    #basepic = "D:\\Arbeit\\V32\\Bildaufnahme"
    forces = pd.read_csv(os.path.join("/media/maxi/T7 Shield/Arbeit/V32/Messdaten/images_force_plain","Force.csv"), sep=',', encoding='utf-8')
    PICPATH = pd.read_csv("/media/maxi/T7 Shield/Arbeit/V32/lookup_with_imgPath.csv")

    #UbuntuPath

    basepic = "/media/maxi/T7 Shield/Arbeit/V32/Bildaufnahme"
    forces=forces.sort_values(by="Hubzahl", key=lambda x: np.argsort(index_natsorted(forces["Hubzahl"])))
    forces = forces.drop_duplicates(subset=["Hubzahl"]).reset_index(drop=True)[:-1]
    #PICPATH1=PICPATH[(PICPATH.Hubzahl > 10000) & (PICPATH.Hubzahl < 100000)]
    #PICPATH2 = PICPATH[(PICPATH.Hubzahl > 150000) & (PICPATH.Hubzahl < 300000)]

    #PICPATH_filered = pd.concat([PICPATH1,PICPATH2])


    PICPATH1 = PICPATH[(PICPATH.Hubzahl > 10000) & (PICPATH.Hubzahl < 175000)]
    PICPATH_filered = pd.concat([PICPATH1])

    test = forces.Hubzahl.isin(PICPATH_filered.Hubzahl)


    test = forces.Hubzahl.isin(PICPATH_filered.Hubzahl)
    #test = forces.Hubzahl.isin(PICPATH.Hubzahl)

    tmp = forces[test].reset_index(drop=True)

    del forces, test, PICPATH

    force, ae, a, imgz = [] , [],[],[]

    k=0
    print('IMGz')
    print('Force started')
    for i in tqdm(tmp.File):
        short = tmp[(tmp.File == i)]
        off = short.offset_values.values
        cal = short.cali_Values.values
        hub = short.Hubzahl
        mul = short.mul_values.values


        impath =PICPATH_filered[(PICPATH_filered.Hubzahl.astype(float)==short.Hubzahl.values[0])]
        impath = impath["IMG_PATH"].to_list()
        impath = impath[0].replace('\\','/')
        im = cv2.imread(os.path.join(basepic,impath),0)
        d = cv2.resize(im, (256,256))
        imgz.append(d.flatten())

    np.savez_compressed("im_vo_filtered.npz", im=imgz)
    del imgz, im

    print('Force started')
    for i in tqdm(tmp.File):
        short = tmp[(tmp.File == i)]
        off = short.offset_values.values
        cal = short.cali_Values.values
        hub = short.Hubzahl
        mul = short.mul_values.values


        i=os.path.join('/media/maxi/T7 Shield',i.replace('\\','/')[3:])
        data = pd.read_csv((i), sep='\t')



        i_forces = (data["Force F"].to_numpy() - off) * cal * mul
        #i_ae = data["Koerperschall"].to_numpy()[2800:8100]
        #i_a = data["Acc a"].to_numpy()

        force.append(i_forces[2800:8100])
        #ae.append(i_ae)
        #a.append(i_a)

    np.savez_compressed("force_vo_filtered.npz", f=force)
    del force, i_forces

    print('ACC started')
    for i in tqdm(tmp.File):
        i=os.path.join('/media/maxi/T7 Shield',i.replace('\\','/')[3:])
        data = pd.read_csv((i), sep='\t')
        short = tmp[(tmp.File == i)]
        off = short.offset_values.values
        cal = short.cali_Values.values
        hub = short.Hubzahl
        mul = short.mul_values.values

        #i_forces = (data["Force F"].to_numpy() - off) * cal * mul
        #i_ae = data["Koerperschall"].to_numpy()
        i_a = data["Acc a"].to_numpy()[2800:8100]

        #force.append(i_forces[2800:8100])
        #ae.append(i_ae[2800:8100])
        a.append(i_a)

    np.savez_compressed("a_vo_filtered.npz", acc=a)
    del a
    del i_a

    print('AE started')
    for i in tqdm(tmp.File):
        i=os.path.join('/media/maxi/T7 Shield',i.replace('\\','/')[3:])
        data = pd.read_csv((i), sep='\t')
        short = tmp[(tmp.File == i)]
        off = short.offset_values.values
        cal = short.cali_Values.values
        hub = short.Hubzahl
        mul = short.mul_values.values

        #i_forces = (data["Force F"].to_numpy() - off) * cal * mul
        i_ae = data["Koerperschall"].to_numpy()[2800:8100]
        #i_a = data["Acc a"].to_numpy()

        #force.append(i_forces[2800:8100])
        ae.append(i_ae)
        #a.append(i_a)

    np.savez_compressed("ae_vo_filtered.npz", ae=ae)
    del ae, i_ae


    #ae=np.load("PCA//a.npz")


