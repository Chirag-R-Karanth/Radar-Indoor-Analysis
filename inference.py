import time

import joblib
import numpy as np
import pandas as pd

# 1. Load the model
model = joblib.load("radar_classifier.pkl")


def get_live_points():
    """
    PLACEHOLDER: You must replace this with your radar's API
    (e.g., Serial.read() for TI IWR radars or a specific SDK)
    """
    # Simulate receiving a cluster of points for one frame
    return pd.DataFrame(
        {
            "x": [0.5, 2.0],
            "y": [1.2, 1.2],
            "z": [1.0, 1.1],
            "doppler": [0.2, 3.0],
            "snr": [15, 16],
        }
    )


print("Starting Real-Time Detection...")

try:
    while True:
        # 2. Capture live data
        df_frame = get_live_points()

        if not df_frame.empty:
            # 3. Extract Features (Must match your training logic exactly)
            features = {
                "x_mean": df_frame["x"].mean(),
                "y_mean": df_frame["y"].mean(),
                "z_mean": df_frame["z"].mean(),
                "doppler_mean": df_frame["doppler"].mean(),
                "snr_mean": df_frame["snr"].mean(),
                "x_std": df_frame["x"].std(),
                "y_std": df_frame["y"].std(),
                "z_std": df_frame["z"].std(),
                "doppler_std": df_frame["doppler"].std(),
                "snr_std": df_frame["snr"].std(),
                "num_points": len(df_frame),
            }

            # Convert to DataFrame and fill NaNs (for single-point frames)
            input_df = pd.DataFrame([features]).fillna(0)

            # 4. Predict
            prediction = model.predict(input_df)[0]

            # 5. Display (This is what 'Mam' wants to see)
            print(f">>> CURRENT STATE: {prediction.upper()}", end="\r")

        time.sleep(0.1)  # Adjust based on radar frame rate
except KeyboardInterrupt:
    print("\nStopping...")
