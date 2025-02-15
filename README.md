# MultiModPunch
Benchmarking Data-Driven Approaches in Industrial Punching: A multi-modal Dataset

Dataset: [Zenodo-Link](https://doi.org/10.5281/zenodo.14782454)

## Citation Notice

If you use this dataset, please cite it as follows:

**Author(s):**           Lorenz, M.  
**Title:**             Benchmarking machine learning approaches in Industrial Punching: A multi-modal Dataset   
**Year:**       2025  
**DOI / URL:**       https://doi.org/10.5281/zenodo.14782454  


Example citation (APA format):  
Lorenz, M. (2025). Benchmarking machine learning approaches in Industrial Punching: A multi-modal Dataset (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14782454

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
      Stroke   Thickness             Sensors              Image   Labels
0      10540  500.684299   Sensors\10540.csv   Images\10540.jpg   normal
1      10553  500.991485   Sensors\10553.csv   Images\10553.jpg   normal
2      10562  501.062754   Sensors\10562.csv   Images\10562.jpg   normal
       ...         ...                 ...                ...           ...
9745  149953  506.187240  Sensors\149953.csv  Images\149953.jpg  failure
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
data.plot_thickness(all=False, coil1=True) #all=True whole data, coil1=True only Coil1, coil2=True only Coil2
```

## Plot KDE Thickness
```python
....
data.plot_kde_thickness(all=False, coil1=True)
#all=True whole data, coil1=True only Coil1, coil2=True only Coil2, same=True Coil1+Coil2 in same graph
```

# Dependicies

Python 3.x\
pandas \
numpy \
matplotlib \
PIL 
