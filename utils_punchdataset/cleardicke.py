import os, sys
import numpy as np
import polars as pl
from tqdm import tqdm
import matplotlib.pyplot as plt
from fnmatch import fnmatch


path = "D:\Arbeit\V32\Data_for_Net\dicke.csv"
data = pl.read_csv(path)
#Reduzieren
data=data.filter(data["Hubzahl"]<=175000)
data=data.drop("TimeStemp",'')

sensor = pl.read_csv(os.path.join("D:\\Arbeit\\V32\\Messdaten\\images_force_plain","Force.csv"))
sensor=sensor.unique(subset=["Hubzahl"]).sort("Hubzahl")



picpath = pl.read_csv("D:\\Arbeit\\V32\\lookup_with_imgPath.csv")

test=sensor["Hubzahl"].is_in(picpath["Hubzahl"])

sensor=sensor.filter(test)
sensor=sensor.filter(sensor["Hubzahl"]<=175000)

for i in sensor["File"].to_list():
    short=sensor.filter(sensor["File"]==i)
    tmp=pl.read_csv(i, separator='\t')
    print(i)
    off = short["offset_values"].item()
    cal = short["cali_Values"].item()
    hub = short["Hubzahl"].item()
    mul = short["mul_values"].item()

    forces = (tmp["Force F"]- off) * cal * mul
    tmp=tmp.drop("masterDegree","Versuchsnummer","Zeit2","Produktiongeschwindigkeit","Production","Zeit1","MX410B_CH 1","AbsZeit1","Hubzahl")
    tmp=tmp[2800:8100]
    tmp.replace("Force F",((tmp["Force F"]- off) * cal * mul))


    #filename
    filename=(i.split("."))[0].split("_")[-1]
    tmp.write_csv(os.path.join("D:\\Arbeit\\V32\\punch_dataset\\Sensors", str(filename)+".csv"))

basepath="D:\\Arbeit\\V32\\Bildaufnahme"
savepath="D:\\Arbeit\\V32\\punch_dataset\\Images"
for i in sensor["Hubzahl"].to_list():
    tmppic=picpath.filter(picpath["Hubzahl"] == int(i))["IMG_PATH"].item()
    cmd=('copy ' + os.path.join(basepath,tmppic) +' '+ os.path.join(savepath, str(int(i))+ ".jpg"))
    os.system(cmd)

root = "D:\\Arbeit\\V32\\Dickenmessung"
pattern = "*.jpg"
imglist = []

for path, subdirs, files in os.walk(savepath):
    for name in files:
        if fnmatch(name, pattern):
            imglist.append(os.path.join("Images", name))


root = "D:\\Arbeit\\V32\\punch_dataset\\Sensors"
pattern = "*.csv"
sensorlist = []

for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            sensorlist.append(os.path.join("Sensors\\", name))

# data=data.with_columns(pl.lit(0).alias('Sensors'))
# data=data.with_columns(pl.lit(0).alias('Images'))
# sensorpath=[]
# imagepath=[]
# for i in data["Hubzahl"].to_list():
#     for name in sensorlist:
#         #(name, "*"+str(i)+".csv")
#         if fnmatch(name, "*\\"+str(i)+".csv"):
#             sensorpath.append(name)
#     for name in imglist:
#         if fnmatch(name, "*\\"+str(i)+".jpg"):
#             imagepath.append(name)



sensorpath=[]
imagepath=[]
for i in data["Hubzahl"].to_list():
    sensorpath.append(os.path.join("Sensors",str(i)+".csv"))
    imagepath.append(os.path.join("Images", str(i) + ".jpg"))

tmp=pl.DataFrame({"Sensors": sensorlist, "Sensors": sensorlist})
data=data.with_columns(pl.Series(sensorpath).alias("Sensors"))
data=data.with_columns(pl.Series(imagepath).alias("Image"))

test1=data["Image"].is_in(imglist)
test2=data["Sensors"].is_in(sensorlist)

test1.value_counts()
test2.value_counts()

data=data.filter(test1)

data.write_csv(os.path.join("D:\\Arbeit\\V32\\punch_dataset","LookUp.csv"))

###names to englisch

root = "D:\\Arbeit\\V32\\punch_dataset\\Sensors"
pattern = "*.csv"
imglist = []

for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            imglist.append(os.path.join(root, name))

for i in tqdm(imglist):
    test=pl.read_csv(i)
    test=test.rename({"Koerperschall":"AE","Acc a":"Acc","Force F":"Force"})
    test.write_csv(i)