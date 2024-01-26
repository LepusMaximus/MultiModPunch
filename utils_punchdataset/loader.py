import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

class punch_data:
    """
    Dataset class

    :param path: path to lookup Table.
    """
    def __init__(self, path):
        self.base_path = path.split("\\LookUp.csv")[0]
        self.data=pd.read_csv(path)


    def loadimage(self, stroke):
        self.im=Image.open(os.path.join(self.base_path,self.data["Image"][self.data["Stroke"]==stroke].values[0]))

    def loadsensordata(self, stroke):
        self.sensor=pd.read_csv(os.path.join(self.base_path, self.data["Sensors"][self.data["Stroke"]==stroke].values[0]))

    def loadforce(self, stroke):
        self.loadsensordata(stroke)
        #print(self.sensor["Force F"])
        return self.sensor["Force"].to_numpy()

    def loadae(self,stroke):
        self.loadsensordata(stroke)
        return self.sensor["AE"].to_numpy()

    def split_ratio(self, frac):
        train = self.data.sample(frac=frac)
        test = self.data.drop(train.index).reset_index(drop=True)
        train=train.reset_index(drop=True)
        return train, test

    def split_contiunius(self, frac):
        lendata=len(self.data)
        tick = int(lendata*frac)
        train = self.data[:tick].reset_index(drop=True)
        test = self.data[tick:].reset_index(drop=True)
        return train,test

    def plot_label(self):
        counts=self.data.Labels.value_counts()
        counts.plot(kind="bar")
        plt.tight_layout()
        plt.show()

    def plot_thickness(self, all=True, coil1=False, coil2=False):
        if all:
            plt.title("Coil1+Coil2")
            plt.plot(self.data.Thickness)
        elif coil1:
            plt.title("Coil1")
            mask=self.data.Stroke <= 78840
            plt.plot(self.data[mask].Thickness)
        elif coil2:
            plt.title("Coil2")
            mask=self.data.Stroke >= 78840
            plt.plot(self.data[mask].Thickness)
        plt.xlabel("Stroke")
        plt.ylabel("Thickness")
        plt.show()

    def plot_kde_thickness(self, all=True, coil1=False, coil2=False, same=False):
        if all:
            plt.title("Coil1+Coil2")
            self.data.Thickness.plot.kde()
        elif coil1:
            plt.title("Coil1")
            mask=self.data.Stroke <= 78840
            self.data[mask].Thickness.plot.kde()
        elif coil2:
            plt.title("Coil2")
            mask=self.data.Stroke >= 78840
            self.data[mask].Thickness.plot.kde()
        elif same:
            plt.title("KDE per coil")
            mask=self.data.Stroke <= 78840
            self.data[mask].Thickness.plot.kde(label="Coil 1")
            mask=self.data.Stroke >= 78840
            self.data[mask].Thickness.plot.kde(label="Coil 2")

            plt.legend()
        plt.xlabel("Thickness")
        plt.show()