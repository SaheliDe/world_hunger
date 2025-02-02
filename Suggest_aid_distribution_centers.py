import folium
import numpy as np
import pandas as pd
from geopy.distance import great_circle
from geopy.point import Point
from sklearn.cluster import DBSCAN


def filter_dataset(file_path, lon_min, lon_max, lat_min, lat_max):
    """
    Filters the dataset based on the specified longitude and latitude range.

    Parameters:
        file_path (str): Path to the CSV file containing the dataset.
        lon_min (float): Minimum longitude value.
        lon_max (float): Maximum longitude value.
        lat_min (float): Minimum latitude value.
        lat_max (float): Maximum latitude value.

    Returns:
        pd.DataFrame: Filtered dataset as a DataFrame.
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Filter based on the specified range
    filtered_data = data[
        (data["longitude"] >= lon_min)
        & (data["longitude"] <= lon_max)
        & (data["latitude"] >= lat_min)
        & (data["latitude"] <= lat_max)
    ]

    return filtered_data


def find_clusters(
    data,
    eps_km=30,
    min_samples=5,
    metric="haversine",
    algorithm="ball_tree",
    leaf_size=30,
    p=None,
):
    coords = data[["latitude", "longitude"]].values
    # Convert epsilon from kilometers to radians
    kms_per_radian = 6371.0088  # Radius of the Earth in kilometers
    eps_rad = eps_km / kms_per_radian

    # Perform DBSCAN clustering
    db = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric=metric,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
    ).fit(np.radians(coords))
    cluster_labels = db.labels_
    data["cluster"] = cluster_labels

    clusters = data[data["cluster"] != -1].groupby("cluster")

    # Find the central points of the clusters
    central_points = []
    for cluster_id, cluster_data in clusters:
        centroid = cluster_data[["latitude", "longitude"]].mean().values
        central_points.append(Point(centroid[0], centroid[1]))

    return central_points


def create_map(central_points):
    # Initialize the map centered on Nigeria
    m = folium.Map(location=[9.0820, 8.6753], zoom_start=6)

    # Add the cluster centers as red crosses
    for point in central_points:
        folium.Marker(
            location=[point.latitude, point.longitude],
            icon=folium.Icon(color="red", icon="plus", prefix="fa"),
        ).add_to(m)

    return m


# def create_map(central_points):
#     # Initialize the map centered on Nigeria
#     m = folium.Map(location=[9.0820, 8.6753], zoom_start=6)

#     # Add the cluster centers as red dots
#     for point in central_points:
#         folium.CircleMarker(
#             location=[point.latitude, point.longitude],
#             radius=5,
#             color="red",
#             fill=True,
#             fill_color="red",
#             fill_opacity=0.6,
#         ).add_to(m)

#     return m


def suggest_distribution_centers(
    file_path,
    coord1,
    coord2,
    eps_km=30,
    min_samples=5,
    metric="haversine",
    algorithm="ball_tree",
    leaf_size=30,
    p=None,
):
    lon_min = min(coord1[1], coord2[1])
    lon_max = max(coord1[1], coord2[1])
    lat_min = min(coord1[0], coord2[0])
    lat_max = max(coord1[0], coord2[0])
    print("Filtering relevant area...")
    filtered_data = filter_dataset(file_path, lon_min, lon_max, lat_min, lat_max)
    print("Creating clusters...")
    central_points = find_clusters(
        filtered_data, eps_km, min_samples, metric, algorithm, leaf_size, p
    )
    print("Creating map...")
    m = create_map(central_points)
    m.save("map.html")
    return m


file_path = "nga_general_2020.csv"  # Replace with the actual file path

m = suggest_distribution_centers(
    file_path=file_path,
    coord1=[13.195500649433004, 7.589886763448066],
    coord2=[12.389669979879995, 8.798102534423935],
    eps_km=1,
    min_samples=5,
)
