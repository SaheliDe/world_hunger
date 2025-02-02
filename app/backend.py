import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from geopy.point import Point
from sklearn.cluster import DBSCAN

app = Flask(__name__)


def filter_dataset(file_path, lon_min, lon_max, lat_min, lat_max):
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


def first_clustering(data, eps_km=3, min_samples=5):
    coords = data[["latitude", "longitude"]].values
    kms_per_radian = 6371.0088
    eps_rad = eps_km / kms_per_radian
    db = DBSCAN(
        eps=eps_rad, min_samples=min_samples, metric="haversine", algorithm="ball_tree"
    ).fit(np.radians(coords))
    cluster_labels = db.labels_
    data["cluster"] = cluster_labels
    clusters = data[data["cluster"] != -1].groupby("cluster")
    central_points = []
    for cluster_id, cluster_data in clusters:
        centroid = cluster_data[["latitude", "longitude"]].mean().values
        central_points.append(Point(centroid[0], centroid[1]))
    return central_points


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    data = request.json
    coord1 = data["coord1"]
    coord2 = data["coord2"]
    print(f"Received coordinates: {coord1}, {coord2}")  # Debugging line

    file_path = "app/nga_general_2020.csv"  # Replace with the actual file path
    lon_min = min(coord1["lng"], coord2["lng"])
    lon_max = max(coord1["lng"], coord2["lng"])
    lat_min = min(coord1["lat"], coord2["lat"])
    lat_max = max(coord1["lat"], coord2["lat"])

    print(
        f"Filtering data within bounds: ({lat_min}, {lon_min}) to ({lat_max}, {lon_max})"
    )  # Debugging line

    filtered_data = filter_dataset(file_path, lon_min, lon_max, lat_min, lat_max)
    central_points = first_clustering(filtered_data)

    print(
        f"Cluster centers: {[{'lat': point.latitude, 'lng': point.longitude} for point in central_points]}"
    )  # Debugging line

    return jsonify(
        [{"lat": point.latitude, "lng": point.longitude} for point in central_points]
    )


if __name__ == "__main__":
    app.run(debug=True)
