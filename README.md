# MultiModPunch
Benchmarking Data-Driven Approaches in Industrial Punching: A multi-modal Dataset


## Load LookUp
```python
import loader
path = "path_to_lookup"
data=loader.punch_data(path)
```
## Access Data
```python
import loader
path = "path_to_lookup"
data=loader.punch_data(path)

data.data
```
Output:
```
       Stroke   Thickness             Sensors              Image        Labels
0        5852  500.768070    Sensors\5852.csv    Images\5852.jpg  initial_wear
1        5865  501.223980    Sensors\5865.csv    Images\5865.jpg  initial_wear
2        5878  500.560970    Sensors\5878.csv    Images\5878.jpg  initial_wear
       ...         ...                 ...                ...           ...
11920  174990  502.915710  Sensors\174990.csv  Images\174990.jpg       failure
```
Acces any Column with:
```python
data.data["Column Name"]
```
## Plot distrubtion of Labels
```python
....

data.plot_label()
```
## Load Image
```python
....

data.loadimage(stroke)
# Acces Image
data.im
```
## Load Force
```python
....
force=data.loadfore(stroke) #as numpy array
```

## Load AE
```python
....
ae=data.loadae(stroke) #as numpy array
```
## Split data in train and test
```python
....
train,test=data.split_contiunius(0.1)
```


## Plot Thickness
```python
....
data.plot_Thickness(all=False, coil1=True) #all=True whole data, coil1=True only Coil1, coil2=True only Coil2
```

## Plot KDE Thickness
```python
....
data.plot_kde_Thickness(all=False, coil1=True)
#all=True whole data, coil1=True only Coil1, coil2=True only Coil2, same=True Coil1+Coil2 in same graph
```

# Dependicies

Python 3.x\
pandas \
numpy \
matplotlib \
PIL 