import glob
import os

import joblib
import matplotlib
import pandas as pd

# Force the backend before importing pyplot
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


def load_data(
    base_path="/home/neo_phantom_byte/Documents/Radar-Indoor-Analysis/DATA SET/ai ml",
):
    all_data = []

    if not os.path.exists(base_path):
        print(f"Error: Path not found: {base_path}")
        return None

    subdirectories = [
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ]
    print(f"Found labels: {subdirectories}")

    for label in subdirectories:
        label_path = os.path.join(base_path, label)
        csv_files = glob.glob(os.path.join(label_path, "*.csv"))

        print(f"Loading {len(csv_files)} files for label: {label}")

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Ensure the file is not empty
                if not df.empty:
                    df["label"] = label
                    # Ensure 'file' column exists for grouping later
                    if "file" not in df.columns:
                        df["file"] = os.path.basename(csv_file)
                    all_data.append(df)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")

    if not all_data:
        print("No data loaded. Check your file paths and CSV content.")
        return None

    return pd.concat(all_data, ignore_index=True)


def preprocess_data(df):
    df = df.copy()

    # Drop columns that have zero variance or are markers
    for col in ["noise", "track_id"]:
        if col in df.columns and df[col].nunique() <= 1:
            df = df.drop(columns=[col])

    # Group by file + frame to create statistical snapshots
    grouped = df.groupby(["file", "frame", "label"])

    features = []

    for (file, frame, label), group in grouped:
        # Standard deviation is NaN if there's only 1 point in a frame.
        # We fill these with 0 later to avoid the 'box_aspect' plotting crash.
        feature_vector = [
            group["x"].mean(),
            group["y"].mean(),
            group["z"].mean(),
            group["doppler"].mean(),
            group["snr"].mean(),
            group["x"].std(),
            group["y"].std(),
            group["z"].std(),
            group["doppler"].std(),
            group["snr"].std(),
            len(group),  # number of points in frame
        ]
        features.append(feature_vector + [label])

    columns = [
        "x_mean",
        "y_mean",
        "z_mean",
        "doppler_mean",
        "snr_mean",
        "x_std",
        "y_std",
        "z_std",
        "doppler_std",
        "snr_std",
        "num_points",
        "label",
    ]

    new_df = pd.DataFrame(features, columns=columns)

    # CRITICAL: Fix for the 'box_aspect' error.
    # Standard deviation of a single point is NaN. Plotting NaN causes the crash.
    new_df = new_df.fillna(0)

    X = new_df.drop("label", axis=1)
    y = new_df["label"]

    return X, y


def train_model(X, y):
    # Ensure we have more than one class to train on
    if len(y.unique()) < 2:
        raise ValueError(
            "Need at least 2 different labels in the dataset to train a classifier."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test


def plot_feature_importance(model, X):
    importances = model.feature_importances_
    feature_names = X.columns

    df_imp = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)

    print("\nFeature Importance:\n", df_imp)

    # Check if sum of importances is 0 to avoid plot errors
    if df_imp["importance"].sum() <= 0:
        print("Skipping Feature Importance plot: All importances are zero.")
        return

    plt.figure(figsize=(10, 6))
    plt.barh(df_imp["feature"], df_imp["importance"], color="skyblue")
    plt.gca().invert_yaxis()
    plt.title("Feature Importance")
    # Using subplots_adjust instead of tight_layout to be safer on Fedora/Wayland
    plt.subplots_adjust(left=0.3)
    plt.show()


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\n=== RESULTS ===")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n")
    print(report)

    # CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues", xticks_rotation=45, ax=plt.gca())

    plt.title("Confusion Matrix")
    plt.subplots_adjust(bottom=0.2)
    plt.show()


if __name__ == "__main__":
    print("Starting Radar Analysis...")
    df_raw = load_data()

    if df_raw is not None:
        print("Total raw data rows:", len(df_raw))

        X, y = preprocess_data(df_raw)

        print("\nX shape (processed frames):", X.shape)
        print("y distribution:\n", y.value_counts())

    if __name__ == "__main__":
        print("Starting Radar Analysis...")
        df_raw = load_data()

        if df_raw is not None:
            X, y = preprocess_data(df_raw)

            try:
                # 1. Train the model
                model, X_test, y_test = train_model(X, y)

                # 2. SAVE THE MODEL HERE (Inside the block where 'model' exists)
                joblib.dump(model, "radar_classifier.pkl")
                print("\nSUCCESS: Model saved as radar_classifier.pkl")

                # 3. Run evaluation
                plot_feature_importance(model, X)
                evaluate_model(model, X_test, y_test)

            except Exception as e:
                print(f"Error during model training/evaluation: {e}")
