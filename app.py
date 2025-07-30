from flask import Flask, render_template, request,jsonify
import pickle
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import numpy as np
import joblib
from datetime import datetime, timedelta
import time
app = Flask(__name__)

# Mapping form value (1-10) to actual place names
place_mapping = {
    "1": "Sri Venkateswara Temple",
    "2": "Sri Padmavathi Temple",
    "3": "Kapila Theertham",
    "4": "Govindaraja Swamy Temple",
    "5": "Sri Kalahasti Temple",
    "6": "Kanipakam Temple",
    "7": "Sri Venkateswara Zoological Park",
    "8": "Sri Vari Museum",
    "9": "Alivelu Mangapuram",
    "10": "Srinivasa Mangapuram" 
}

# Load Excel for backup (optional)
df_full = pd.read_excel("Tirumala_Tourist_Places_53.xlsx")

# Load precomputed nearest neighbors dict
with open("nearest_neighbors.pkl", "rb") as f:
    nearest_neighbors = pickle.load(f)


# Load model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load historical data
df = pd.read_csv("unpreprocesseddata.csv")
df2 = pd.read_csv("your_data.csv")  # historical timestamp + pilgrim data

def get_festival_impact(festival_name, df):
    global_mean = df['Piligrims'].mean()
    festival_means = df[df['Special Day'] != 'normal'].groupby('Special Day')['Piligrims'].mean()

    smoothing_factor = 0.2
    if festival_name in festival_means:
        raw_mean = festival_means[festival_name]
        smoothed_impact = (1 - smoothing_factor) * raw_mean + smoothing_factor * global_mean
        return smoothed_impact
    else:
        return global_mean


@app.route("/", methods=["GET","POST"])
def home():
        
    return render_template("WebPage.html")

@app.route('/index')
def index():
    today_str = datetime.today().strftime('%Y-%m-%d')
    return render_template('index.html', default_date=today_str)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    future_date_str = data['date']
    is_weekend = int(data['is_weekend'])
    is_holiday = int(data['is_holiday'])
    festival_name = data['festival_name']

    try:
        future_date = datetime.strptime(future_date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400

    timestamp = int(time.mktime(future_date.timetuple()))
    impact_value = get_festival_impact(festival_name, df)

    # Use 2024's data for 7-day avg
    date_2024 = future_date.replace(year=2024)
    start = int(time.mktime((date_2024 - timedelta(days=7)).timetuple()))
    end = int(time.mktime(date_2024.timetuple()))

    past_7_days = df2[(df2["Timestamp"] >= start) & (df2["Timestamp"] < end)]
    pilgrims_7day_avg = past_7_days["Piligrims"].mean()

    if np.isnan(pilgrims_7day_avg):
        pilgrims_7day_avg = df2["Piligrims"].mean()

    if is_holiday:
        pilgrims_7day_avg *= 1.0336

    new_data = {
        'Timestamp': timestamp,
        'weekday': is_weekend,
        'festival_impact': impact_value,
        'Piligrims_7Day_Avg': pilgrims_7day_avg
    }

    new_data_df = pd.DataFrame([new_data])
    X_new_scaled = scaler.transform(new_data_df)
    prediction = model.predict(X_new_scaled)

    return jsonify({'prediction': int(prediction[0])})



@app.route("/nearby", methods=["GET", "POST"])
def nearby():
    results = []

    if request.method == "POST":
        selected_index = request.form.get("base_place")
        max_distance = float(request.form.get("distance"))

        # Convert selected index to place name
        base_place = place_mapping.get(selected_index)

        # Load nearby places for selected base
        all_neighbors = nearest_neighbors.get(base_place, pd.DataFrame())

        # Filter by max distance
        nearby = all_neighbors[all_neighbors["Distance (km)"] <= max_distance].copy()
        nearby["Rank"] = range(1, len(nearby) + 1)

        results = nearby.to_dict(orient="records")

        # Base coordinates for map centering
        if not nearby.empty:
            base_lat = all_neighbors.iloc[0]["Latitude"]
            base_lon = all_neighbors.iloc[0]["Longitude"]
        else:
            base_lat, base_lon = 13.6839, 79.3475  # fallback to Tirupati

        # Create Folium map
        m = folium.Map(location=[base_lat, base_lon], zoom_start=12)
        marker_cluster = MarkerCluster().add_to(m)

        # Add base location marker
        folium.Marker(
            [base_lat, base_lon],
            popup=f"Base: {base_place}",
            icon=folium.Icon(color="red")
        ).add_to(m)

        for _, row in nearby.iterrows():
            folium.Marker(
                [row["Latitude"], row["Longitude"]],
                popup=f"{row['Place Name']} ({row['Distance (km)']:.2f} km)",
                icon=folium.Icon(color="blue")
            ).add_to(marker_cluster)

            folium.PolyLine(
                [(base_lat, base_lon), (row["Latitude"], row["Longitude"])],
                color='blue'
            ).add_to(m)


        # Save the map
        m.save("static/recommended_map.html")

    return render_template("near_places.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
