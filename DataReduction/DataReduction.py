from sklearn.manifold import TSNE
import umap
import gudhi as gd
import gudhi.representations
import numpy as np


# Zusammenfassung der Dimensionen:
#
#     t-SNE: Empfohlene Reduktion auf 10 Dimensionen (stärkere Einschränkung bei höheren Dimensionen).
#     UMAP: Empfohlene Reduktion auf 15 Dimensionen (besser geeignet für höhere Dimensionen).
#     PaCHAP: Empfohlene Reduktion auf 20 Dimensionen (mit hierarchischer PCA als Platzhalter).
#     TI: Empfohlene Reduktion auf 25 Dimensionen (unter Verwendung von Persistent Homology).


def reduceTSNE(dims, datapath):
    # t-SNE für Bilddaten (auf z.B. 10 Dimensionen reduzieren)
    tsne_image = TSNE(n_components=dims, random_state=42)
    reduced_image_data = tsne_image.fit_transform(datapath)

def reduceUMAP(dims, datapath):
    # UMAP für Bilddaten (auf z.B. 15 Dimensionen reduzieren)
    umap_image = umap.UMAP(n_components=dims, random_state=42)
    reduced_image_data = umap_image.fit_transform(datapath)

def reduceTITS(time_series_data):
    # Persistent Homology für Zeitreihen
    # Create a Vietoris-Rips complex from time series data
    rips_complex = gd.RipsComplex(points=time_series_data, max_edge_length=2.0)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

    # Berechne Persistence Diagram
    diag = simplex_tree.persistence()

    # Für persistente Reduktion (z.B. Dimension 25 aus Diagram)
    persistence_diagram = gd.representations.PersistenceLandscape(resolution=1000)
    reduced_time_series_data = persistence_diagram.fit_transform([diag])

def reduceTIIMG(image_data):
    persistence_diagram = gd.representations.PersistenceLandscape(resolution=1000)
    # Persistent Homology für Bilddaten
    rips_complex_image = gd.RipsComplex(points=image_data, max_edge_length=2.0)
    simplex_tree_image = rips_complex_image.create_simplex_tree(max_dimension=2)
    diag_image = simplex_tree_image.persistence()
    reduced_image_data = persistence_diagram.fit_transform([diag_image])