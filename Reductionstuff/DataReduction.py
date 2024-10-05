from sklearn.manifold import TSNE
import umap
import gudhi as gd
import gudhi.representations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.metrics import silhouette_score
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.manifold import trustworthiness
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from tqdm import tqdm
import pacmap
from scipy.stats import pearsonr
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler



# Zusammenfassung der Dimensionen:
#
#     t-SNE: Empfohlene Reduktion auf 10 Dimensionen (stärkere Einschränkung bei höheren Dimensionen).
#     UMAP: Empfohlene Reduktion auf 15 Dimensionen (besser geeignet für höhere Dimensionen).
#     PaCHAP: Empfohlene Reduktion auf 20 Dimensionen (mit hierarchischer PCA als Platzhalter).
#     TI: Empfohlene Reduktion auf 25 Dimensionen (unter Verwendung von Persistent Homology).


def prepare_data(base_dir, image_dir=None, time_series_dir=None, mode=None):
    """
    Lädt und bereitet Bild- und Zeitreihendaten vor.

    Parameters:
        base_dir (str): base_path for Images and TimeSeries
        image_dir (str): list of subpath to images.
        time_series_dir (str): Pfad zum Verzeichnis der Zeitreihen.
        mode (str): Force or AE

    Returns:
        image_data (np.ndarray): Array der Bilddaten (100, 1310720).
        time_series_data (np.ndarray): Array der Zeitreihendaten (100, 5200).
        labels (np.ndarray): Array der Labels (100,).
    """
    image_data=None
    time_series_data=None

    if image_dir is not None:
        # Bilddaten laden
        print('Processing Images')
        image_data = []
        for img_file in tqdm(image_dir):  # Nur die ersten 100 Bilder
            img_path = os.path.join(base_dir, img_file.replace('\\', '/'))
            img = Image.open(img_path).convert('L')  # Graustufen
            img = img.resize((256,256))
            image_array = np.array(img)
            normalized_array = image_array / 255.0
            img_array = normalized_array.reshape(-1)  # In 1D-Vektor umwandeln
            del normalized_array
            del image_array
            image_data.append(img_array)
        image_data = np.array(image_data)
        print(f"Shape of image data: {image_data.shape}")  # (100, 1310720)

        # Daten skalieren
        #scaler_image = Normalizer()
        #image_data = scaler_image.fit_transform(image_data)

    if time_series_dir is not None:
        # Zeitreihendaten laden
        time_series_data = []
        print(f'Processing TimeSeries {mode}')
        for ts_file in tqdm(time_series_dir):  # Nur die ersten 100 Zeitreihen
            ts_path = os.path.join(base_dir, ts_file.replace('\\', '/'))
            data = pd.read_csv((ts_path))
            if mode == 'AE':
                i_AE = data["AE"]
                time_series_data.append(i_AE.to_numpy())
            if mode == 'Force':
                i_forces = data["Force"]
                time_series_data.append(i_forces.to_numpy())
        time_series_data = np.array(time_series_data)
        print(f"Shape of time series data: {time_series_data.shape}")  # (100, 5200)

        scaler_time = Normalizer()
        time_series_data = scaler_time.fit_transform(time_series_data)


    return image_data, time_series_data


# def reduce_dimensions(method, data, n_components):
#     """
#     Führt Dimensionalitätsreduktion auf den gegebenen Daten durch.
#
#     Parameters:
#         method (str): Die Reduktionsmethode ('tsne', 'umap', 'pachap', 'ti').
#         data (np.ndarray): Die Daten, die reduziert werden sollen.
#         n_components (int): Die Ziel-Dimension nach der Reduktion.
#
#     Returns:
#         reduced_data (np.ndarray): Die reduzierten Daten.
#     """
#     if method.lower() == 'tsne':
#         if n_components <= 3:
#             reducer = TSNE(n_components=n_components, random_state=42)
#             reduced_data = reducer.fit_transform(data)
#         else:
#             raise ValueError("Unbekannte Reduktionsmethode. Wähle 'tsne', 'umap', 'pachap' oder 'ti'.")
#
#     elif method.lower() == 'umap':
#         reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=n_components, random_state=42, metric='euclidean', verbose=True)
#         reduced_data = reducer.fit_transform(data)
#
#     elif method.lower() == 'pachap':
#         # PaCHAP ist nicht standardmäßig verfügbar, daher verwenden wir hier PCA als Platzhalter
#         # Implementiere hier eine spezifische PaCHAP-Logik, falls verfügbar
#         # Erstelle ein PaCMAP-Objekt
#         reducer = pacmap.PaCMAP(n_components=n_components, random_state=42)
#
#         # Reduziere die Dimensionen
#         reduced_data = reducer.fit_transform(data)
#
#
#     elif method.lower() == 'imgti':
#         persistence_diagram = gd.representations.PersistenceLandscape(resolution=1000)
#         # Persistent Homology für Bilddaten
#         rips_complex_image = gd.RipsComplex(points=data, max_edge_length=2.0)
#         simplex_tree_image = rips_complex_image.create_simplex_tree(max_dimension=2)
#         diag_image = simplex_tree_image.persistence()
#         reduced_image_data = persistence_diagram.fit_transform([diag_image])
#
#     elif method.lower() == 'timeti':
#         # Topologische Interaktion (Ti) mit Gudhi - Beispiel für Persistent Homology
#         rips_complex = gd.RipsComplex(points=data, max_edge_length=2.0)
#         simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
#         diag = simplex_tree.persistence()
#         # Verwende Persistence Landscape als repräsentative Features
#         persistence_landscape = gd.representations.PersistenceLandscape(resolution=1000)
#         reduced_data = persistence_landscape.fit_transform([diag])[0]
#         # Skalierung, falls notwendig
#         scaler = StandardScaler()
#         reduced_data = scaler.fit_transform(reduced_data)
#
#
#     else:
#         raise ValueError("Unbekannte Reduktionsmethode. Wähle 'tsne', 'umap', 'pachap' oder 'ti'.")
#
#     return reduced_data

def load_data(methode):
    """
    Load the reduced data.

    Parameters:
        methode(str): img, force, AE

    Returns:
        data
    """

    if methode.lower() == 'img':
        data=np.load("/media/maxi/T7 Shield/Arbeit/AutoEncoder/Data_reduced/image_reduced_1.npy", allow_pickle=True)
    elif methode.lower() == 'force':
        data = np.load("/media/maxi/T7 Shield/Arbeit/AutoEncoder/Data_reduced/time_reduced_Force.npy", allow_pickle=True)
    elif methode.lower() == 'ae':
        data = np.load("/media/maxi/T7 Shield/Arbeit/AutoEncoder/Data_reduced/time_reduced_AE.npy", allow_pickle=True)
    elif methode.lower() == 'labels':
        data = np.load("/media/maxi/T7 Shield/Arbeit/AutoEncoder/Data_reduced/image_reduced_Lables.npy", allow_pickle=True)
    else:
        raise ValueError("No data. Choose 'img', 'force','AE' or 'Labels'.")

    return data

def load_data_norm(methode):
    """
    Load the reduced data.

    Parameters:
        methode(str): img, force, AE

    Returns:
        data
    """

    if methode.lower() == 'img':
        data=np.load("/media/maxi/HDD/arbeit_auslagerung/UMAP/image_reduced_1_255.npy", allow_pickle=True)
    elif methode.lower() == 'force':
        data = np.load("/media/maxi/HDD/arbeit_auslagerung/UMAP/time_reduced_normForce.npy", allow_pickle=True)
    elif methode.lower() == 'ae':
        data = np.load("/media/maxi/HDD/arbeit_auslagerung/UMAP/time_reduced_normAE.npy", allow_pickle=True)
    elif methode.lower() == 'labels':
        data = np.load("/media/maxi/T7 Shield/Arbeit/AutoEncoder/Data_reduced/image_reduced_Lables.npy", allow_pickle=True)
    else:
        raise ValueError("No data. Choose 'img', 'force','AE' or 'Labels'.")

    return data


def save_data(method, image_reduced=None, time_reduced=None):
    """
    Speichert die Bild- und Zeitreihendaten.

    Parameters:
        method (str): Die Reduktionsmethode ('tsne', 'umap', 'pachap', 'ti').
        image_reduced (np.ndarray): Die reduzierten Bilddaten.
        time_reduced (np.ndarray): Die reduzierten Zeitreihendaten.
    """
    if image_reduced is not None:
        np.save(f'/media/maxi/HDD/arbeit_auslagerung/UMAP/image_reduced_{method}.npy', image_reduced)
    if time_reduced is not None:
        np.save(f'/media/maxi/HDD/arbeit_auslagerung/UMAP/time_reduced_{method}.npy', time_reduced)
    print(f"Daten mit Methode {method} gespeichert.")


def validate_reduction(method, original_data, reduced_data, n_neighbors=5):
    """
    Validiert die Dimensionalitätsreduktion mit Trustworthiness.

    Parameters:
        method (str): Die Reduktionsmethode.
        original_data (np.ndarray): Die originalen Daten.
        reduced_data (np.ndarray): Die reduzierten Daten.
        n_neighbors (int): Anzahl der Nachbarn für die Berechnung.

    Returns:
        trust_score (float): Der Trustworthiness-Score.
    """
    trust_score = trustworthiness(original_data, reduced_data, n_neighbors=n_neighbors)
    print(f"{method.upper()} Trustworthiness: {trust_score:.3f}")
    return trust_score


def calculate_silhouette(method, reduced_data, labels):
    """
    Berechnet den Silhouette-Score für die reduzierten Daten.

    Parameters:
        method (str): Die Reduktionsmethode.
        reduced_data (np.ndarray): Die reduzierten Daten.
        labels (np.ndarray): Die zugehörigen Labels.

    Returns:
        silhouette_avg (float): Der durchschnittliche Silhouette-Score.
    """
    silhouette_avg = silhouette_score(reduced_data, labels)
    print(f"{method.upper()} Silhouette Score: {silhouette_avg:.3f}")
    return silhouette_avg




def calculate_stress(original_data, reduced_data):
    """
    Berechnet den Stress der Reduktion.

    Parameters:
        original_data (np.ndarray): Die originalen hochdimensionalen Daten.
        reduced_data (np.ndarray): Die reduzierten niedrigdimensionalen Daten.

    Returns:
        stress (float): Der Stress-Wert.
    """
    # Berechne paarweise Distanzen
    original_dist = pairwise_distances(original_data)
    reduced_dist = pairwise_distances(reduced_data)

    # Berechne den Stress
    stress = np.sqrt(np.sum((original_dist - reduced_dist) ** 2) / np.sum(original_dist ** 2))
    return stress


from sklearn.metrics import mean_squared_error


def calculate_reconstruction_error(original_data, reconstructed_data, method):
    """
    Berechnet den Rekonstruktionsfehler.

    Parameters:
        original_data (np.ndarray): Die originalen Daten.
        reconstructed_data (np.ndarray): Die rekonstruierten Daten.
        method (str): Die Reduktionsmethode.

    Returns:
        mse (float): Der mittlere quadratische Fehler.
    """
    mse = mean_squared_error(original_data, reconstructed_data)
    print(f"{method.upper()} Reconstruction MSE: {mse:.3f}")
    return mse


def visualize_as_graph(embeddings, labels, n_neighbors=5, method="umap"):
    """
    Visualisiert die reduzierten Embeddings als k-NN-Graph.

    Parameters:
        embeddings (np.ndarray): Die reduzierten Daten (Embeddings).
        labels (np.ndarray): Die zugehörigen Labels (für Färbung der Knoten).
        n_neighbors (int): Anzahl der Nachbarn für den k-NN-Graphen.
        method (str): Die verwendete Reduktionsmethode (nur zur Titelanzeige).
    """
    # Berechne k-NN-Graphen
    knn_graph = kneighbors_graph(embeddings, n_neighbors=n_neighbors, mode='connectivity', include_self=False)

    # Erstelle einen Graphen mit NetworkX
    G = nx.Graph(knn_graph)

    # Zeichne den Graphen
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)  # Berechne Layout des Graphen

    # Färbe die Knoten basierend auf den Labels
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        nodes = [n for n in G.nodes if labels[n] == label]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=[colors[i]] * len(nodes), label=f"Label {label}")

    # Zeichne die Kanten des Graphen
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # Achsen entfernen und Titel setzen
    plt.title(f"{method.upper()} Visualization with k-NN Graph (k={n_neighbors})")
    plt.axis("off")
    plt.legend()
    plt.show()

def reduce_dimensions(method, data, n_components, config):
    """
    Führt Dimensionalitätsreduktion mit der angegebenen Methode durch.

    Parameters:
        method (str): Die Reduktionsmethode ('tsne', 'umap', 'pachap', 'ti', 'pca').
        data (np.ndarray): Die Daten, die reduziert werden sollen.
        n_components (int): Die Ziel-Dimension nach der Reduktion.
        config (dict): Hyperparameter-Konfiguration für die Methode.

    Returns:
        reduced_data (np.ndarray): Die reduzierten Daten.
    """
    if method.lower() == 'tsne':
        # t-SNE unterstützt nur 2 oder 3 Dimensionen
        if n_components not in [2, 3]:
            raise ValueError("t-SNE unterstützt nur 2 oder 3 Dimensionen.")
        tsne = TSNE(
            n_components=n_components,
            perplexity=config.get("perplexity", 30),
            learning_rate=config.get("learning_rate", 200),
            n_iter=config.get("n_iter", 1000),
            random_state=42,
            verbose=1
        )
        reduced_data = tsne.fit_transform(data)

    elif method.lower() == 'umap':
        umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=config.get("n_neighbors", 15),
            min_dist=config.get("min_dist", 0.1),
            metric=config.get("metric", "euclidean"),
            random_state=42,
            verbose=True
        )
        reduced_data = umap_reducer.fit_transform(data)

    elif method.lower() == 'pachap':
        pacmap_reducer = pacmap.PaCMAP(
            n_components=n_components,
            n_neighbors=config.get("n_neighbors", 15),
            num_iters=config.get("num_iters", 500),
            verbose=True,
            random_state=42
        )
        reduced_data = pacmap_reducer.fit_transform(data)

    elif method.lower() == 'ti':
        # Beispielhafte Implementierung für Topologische Interaktion (TI)
        rips_complex = gd.RipsComplex(points=data, max_edge_length=config.get("max_edge_length", 2.0))
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        diag = simplex_tree.persistence()
        persistence_landscape = gd.representations.PersistenceLandscape(resolution=1000)
        reduced_data = persistence_landscape.fit_transform([diag])[0]
        # Normalisieren nach TI
        scaler = Normalizer()
        reduced_data = scaler.fit_transform(reduced_data.reshape(1, -1))[0].reshape(-1, 1)
        # Optional: Anpassung der Dimension
        if n_components > 1:
            pca = PCA(n_components=n_components)
            reduced_data = pca.fit_transform(reduced_data)

    elif method.lower() == 'pca':
        pca = PCA(
            n_components=n_components,
            random_state=42
        )
        reduced_data = pca.fit_transform(data)

    else:
        raise ValueError("Unbekannte Reduktionsmethode. Wähle 'tsne', 'umap', 'pachap', 'ti' oder 'pca'.")

    return reduced_data


def objective(config, data, labels):
    """
    Objective-Funktion für Ray Tune Hyperparameter-Optimierung.

    Parameters:
        config (dict): Konfiguration, die die Methode und ihre Hyperparameter enthält.
        data (np.ndarray): Die zu reduzierenden Daten.
        labels (np.ndarray): Die zugehörigen Labels.

    Reports:
        trustworthiness (float): Trustworthiness Score der Reduktion.
        silhouette (float): Silhouette Score der Reduktion.
        distance_correlation (float): Korrelationskoeffizient der paarweisen Distanzen.
    """
    method = config["method"]
    n_components = config["n_components"]

    try:
        reduced_data = reduce_dimensions(method, data, n_components, config)
    except ValueError as e:
        # Wenn t-SNE eine ungültige Dimension erhält
        ray.train.report({'trustworthiness':0.0, 'silhouette':0.0, 'distance_correlation':0.0})
        return

    # Berechnung der Metriken
    trust = trustworthiness(data, reduced_data, n_neighbors=5)
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(reduced_data, labels)
    else:
        silhouette = 0.0

    # Pairwise Distance Correlation
    original_dist = pairwise_distances(data).flatten()
    reduced_dist = pairwise_distances(reduced_data).flatten()
    # Entferne Diagonale (Distanzen zu sich selbst sind 0)
    mask = original_dist > 0
    original_dist = original_dist[mask]
    reduced_dist = reduced_dist[mask]
    correlation, _ = pearsonr(original_dist, reduced_dist)

    # Report an Ray Tune
    ray.train.report({'trustworthiness':trust, 'silhouette':silhouette, 'distance_correlation':correlation})




def get_search_space():
    """
    Definiert den Hyperparameter-Suchraum für alle Reduktionsmethoden.

    Returns:
        dict: Suchraum für Ray Tune.
    """
    search_space = {
        "method": tune.choice(["umap", "tsne", "pachap"]),
        "n_components": tune.choice([2, 3, 5, 10, 15, 20, 25]),
        # Gemeinsame Hyperparameter, werden nur für relevante Methoden genutzt
        "n_neighbors": tune.choice([10, 15, 20, 30, 50]),
        "min_dist": tune.uniform(0.0, 0.5),
        "perplexity": tune.choice([5, 30, 50]),
        "learning_rate": tune.choice([10, 100, 200]),
        "num_iters": tune.choice([100]),
        "max_edge_length": tune.uniform(1.0, 3.0),
        "metric": tune.choice(["euclidean"])
    }
    return search_space



def convert_labels_to_numeric(labels):
    """
    Konvertiert String-Labels in numerische Labels.

    Parameters:
        labels (list): Liste von String-Labels.

    Returns:
        numeric_labels (list): Liste von numerischen Labels.
        label_mapping (dict): Dictionary, das die String-Labels auf numerische Labels abbildet.
    """
    # Erstelle ein Mapping von String-Labels zu numerischen Labels
    unique_labels = list(set(labels))  # Liste von einzigartigen Labels
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    # Wandle die String-Labels in numerische Labels um
    numeric_labels = [label_mapping[label] for label in labels]

    return numeric_labels, label_mapping


def visualize_embeddings_3D_plotly(comp, embeddings, labels, label_mapping, method="umap"):
    """
    Visualisiert 3D-Embeddings in einem interaktiven Plot und speichert diesen als HTML.

    Parameters:
        embeddings (np.ndarray): Die reduzierten Daten (Embeddings).
        labels (list): Liste der numerischen Labels für die Färbung der Punkte.
        label_mapping (dict): Dictionary, das die numerischen Labels auf die ursprünglichen String-Labels abbildet.
        method (str): Die verwendete Reduktionsmethode (nur zur Titelanzeige).
    """
    # Erstelle die Farben für die verschiedenen Labels
    unique_labels = np.unique(labels)
    colors = np.linspace(0, 1, len(unique_labels))

    # Liste für die Trace-Daten (jede Klasse wird separat geplottet)
    data = []

    for i, label in enumerate(unique_labels):
        # Wähle die Datenpunkte mit dem aktuellen Label
        idx = np.where(labels == label)

        # Original-Label für die Legende
        label_name = list(label_mapping.keys())[list(label_mapping.values()).index(label)]

        # Hinzufügen eines Scatter3D-Objekts für diese Klasse
        scatter = go.Scatter3d(
            x=embeddings[idx, 0][0],
            y=embeddings[idx, 1][0],
            z=embeddings[idx, 2][0],
            mode='markers',
            marker=dict(size=8, color=colors[i], colorscale='Rainbow', opacity=1),
            name=label_name
        )
        data.append(scatter)

    # Berechne die minimalen und maximalen Werte der Embeddings für jede Achse
    x_range = [embeddings[:, 0].min(), embeddings[:, 0].max()]
    y_range = [embeddings[:, 1].min(), embeddings[:, 1].max()]
    z_range = [embeddings[:, 2].min(), embeddings[:, 2].max()]

    # Layout für die Achsen und den Titel mit automatischer Skalierung der Achsen
    layout = go.Layout(
        title=f"{method.upper()} 3D Embeddings Visualization, Components={comp}",
        scene=dict(
            xaxis=dict(title='Embedding Dimension 1', range=x_range, autorange=True),
            yaxis=dict(title='Embedding Dimension 2', range=y_range, autorange=True),
            zaxis=dict(title='Embedding Dimension 3', range=z_range, autorange=True),
            aspectmode="cube"  # sorgt für gleichmäßige Skalierung der Achsen
        )
    )

    # Erstellen der Figur
    fig = go.Figure(data=data, layout=layout)

    # Speichern als HTML (dynamisch interaktiv)
    fig.write_html(f'/media/maxi/HDD/arbeit_auslagerung/UMAP/tt_norm_UMAP_3D_{comp}.html')

    # Zeige den Plot interaktiv
    fig.show()




def main():
    # Vorbereitung der Daten
    labels=load_data_norm('labels')
    img=load_data_norm('img')
    force = load_data_norm('force')
    ae = load_data_norm('ae')

    # Konvertiere Labels
    numeric_labels, label_mapping = convert_labels_to_numeric(labels)


    # Definiere den Suchraum
    search_space = get_search_space()

    # Definiere den Scheduler
    scheduler = ASHAScheduler(
        metric="trustworthiness",
        mode="max",
        max_t=100,
        grace_period=10,
        reduction_factor=2
    )

    # Starte Ray Tune
    analysis = tune.run(
        tune.with_parameters(objective, data=img, labels=numeric_labels),
        resources_per_trial={"cpu": 4, "gpu": 0.5},  # Passe an deine Rechnerleistung an
        config=search_space,
        num_samples=50,  # Anzahl der Versuche
        scheduler=scheduler,
        name="dimensionality_reduction_hyperopt_img",
        storage_path="/media/maxi/HDD/arbeit_auslagerung/ray_results",
        log_to_file=True,
        progress_reporter=tune.CLIReporter(
            parameter_columns=["method", "n_components"],
            metric_columns=["trustworthiness", "silhouette", "distance_correlation"],
            max_progress_rows=10,
            sort_by_metric="trustworthiness",
            metric="trustworthiness",
            mode="max"
        )
    )

    # Beste Ergebnisse anzeigen
    print("Best hyperparameters found were: ", analysis.best_config)

    # Exportiere die Ergebnisse für TensorBoard
    df = analysis.results_df
    df.to_csv("/media/maxi/HDD/arbeit_auslagerung/ray_results/dimensionality_reduction_hyperopt/results.csv", index=False)



if __name__ == '__main__':
    main()
    # basepath = "/media/maxi/T7 Shield/Arbeit/V32/punch_dataset/Zenodo"
    # lookup = "/media/maxi/T7 Shield/Arbeit/V32/punch_dataset/Zenodo/LookUp.csv"
    # looki = pd.read_csv(lookup, index_col=0)
    # imglist = looki['Image'].tolist()
    # sensorlist = looki['Sensors'].tolist()
    # #labellist = looki['Labels'].to_numpy()
    # #save_data('Lables', labellist, None)
    #
    # #Imagedata
    # image_data, time_series_data =prepare_data(basepath, imglist, None)
    # save_data('1_255',image_data,None)
    # del image_data
    #
    # #Timeseries for Force and AE
    # image_data, time_series_data = prepare_data(basepath, None, sensorlist, 'Force')
    # save_data('normForce', None, time_series_data)
    # del time_series_data
    # image_data, time_series_data = prepare_data(basepath, None, sensorlist, 'AE')
    # save_data('normAE', None, time_series_data)

    methode='pachap'

    ##IMAGE
    labels=load_data_norm('labels')
    img=load_data_norm('img')
    components = [3,4,6,8,10,12,14,16,18,20,25,30]
    components = [3]
    for i in tqdm(components):
        reddata=reduce_dimensions(methode,img,i)
        numeric_labels, label_mapping=convert_labels_to_numeric(labels)
        print(calculate_silhouette(methode, reddata, labels))
        print(validate_reduction(methode, img, reddata))
        print(calculate_stress(img, reddata))
        #visualeUMAP(reddata, numeric_labels)
        #visualize_embeddings(i,reddata, numeric_labels, label_mapping)
        visualize_embeddings_3D_plotly(i,reddata, numeric_labels, label_mapping, methode)
    #visualize_as_graph(reddata, labels)

    ##TIMESERIERS
    # labels=load_data_norm('labels')
    # force=load_data_norm('force')
    # components = [3,4,6,8,10,12,14,16,18,20,25,30]
    # for i in tqdm(components):
    #     reddata=reduce_dimensions(methode,force,i)
    #     numeric_labels, label_mapping=convert_labels_to_numeric(labels)
    #     #visualeUMAP(reddata, numeric_labels)
    #     #visualize_embeddings(i,reddata, numeric_labels, label_mapping)
    #     visualize_embeddings_3D_plotly(i,reddata, numeric_labels, label_mapping, methode)

    # embeddings=reddata
    # labels=numeric_labels
    # method="umap"
    # # Erstelle die Farben für die verschiedenen Labels
    # unique_labels = np.unique(labels)
    # colors = np.linspace(0, 1, len(unique_labels))
    #
    # # Liste für die Trace-Daten (jede Klasse wird separat geplottet)
    # data = []
    #
    # for i, label in enumerate(unique_labels):
    #     # Wähle die Datenpunkte mit dem aktuellen Label
    #     idx = np.where(labels == label)
    #
    #     # Original-Label für die Legende
    #     label_name = list(label_mapping.keys())[list(label_mapping.values()).index(label)]
    #
    #     # Hinzufügen eines Scatter3D-Objekts für diese Klasse
    #     scatter = go.Scatter3d(
    #         x=embeddings[idx, 0][0],
    #         y=embeddings[idx, 1][0],
    #         z=embeddings[idx, 2][0],
    #         mode='markers',
    #         marker=dict(size=5, color=colors[i], colorscale='Rainbow', opacity=1),
    #         name=label_name
    #     )
    #     data.append(scatter)
    #
    # # Berechne die minimalen und maximalen Werte der Embeddings für jede Achse
    # x_range = [embeddings[:, 0].min(), embeddings[:, 0].max()]
    # y_range = [embeddings[:, 1].min(), embeddings[:, 1].max()]
    # z_range = [embeddings[:, 2].min(), embeddings[:, 2].max()]
    #
    # # Layout für die Achsen und den Titel mit automatischer Skalierung der Achsen
    # layout = go.Layout(
    #     title=f"{method.upper()} 3D Embeddings Visualization",
    #     scene=dict(
    #         xaxis=dict(title='Embedding Dimension 1', range=x_range),
    #         yaxis=dict(title='Embedding Dimension 2', range=y_range),
    #         zaxis=dict(title='Embedding Dimension 3', range=z_range),
    #         aspectmode="cube"  # sorgt für gleichmäßige Skalierung der Achsen
    #     )
    # )
    #
    # # Erstellen der Figur
    # fig = go.Figure(data=data, layout=layout)
    # fig.show()
