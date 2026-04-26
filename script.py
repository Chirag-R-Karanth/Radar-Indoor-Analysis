import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import glob

def load_data(base_path="/home/neo_phantom_byte/Documents/Radar-Indoor-Analysis/DATA SET/ai ml"):
    all_data = []
    
    subdirectories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    print(f"Found labels: {subdirectories}")

    for label in subdirectories:
        label_path = os.path.join(base_path, label)
        csv_files = glob.glob(os.path.join(label_path, "*.csv"))
        
        print(f"Loading {len(csv_files)} files for label: {label}")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['label'] = label
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                
    if not all_data:
        print("No data loaded.")
        return None

    return pd.concat(all_data, ignore_index=True)


def preprocess_data(df):
    df = df.copy()

    # Drop useless columns
    for col in ['noise', 'track_id']:
        if col in df.columns and df[col].nunique() == 1:
            df = df.drop(columns=[col])

    # Group by file + frame
    grouped = df.groupby(['file', 'frame', 'label'])

    features = []

    for (file, frame, label), group in grouped:
        feature_vector = [
            group['x'].mean(),
            group['y'].mean(),
            group['z'].mean(),
            group['doppler'].mean(),
            group['snr'].mean(),

            group['x'].std(),
            group['y'].std(),
            group['z'].std(),
            group['doppler'].std(),
            group['snr'].std(),

            len(group)  # number of points in frame
        ]

        features.append(feature_vector + [label])

    columns = [
        'x_mean','y_mean','z_mean','doppler_mean','snr_mean',
        'x_std','y_std','z_std','doppler_std','snr_std',
        'num_points','label'
    ]

    new_df = pd.DataFrame(features, columns=columns)

    X = new_df.drop('label', axis=1)
    y = new_df['label']

    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    if df is not None:
        print("Total rows:", len(df))
        print(df.head())

        X, y = preprocess_data(df)

        print("\nX shape:", X.shape)
        print("y distribution:\n", y.value_counts())

        model, X_test, y_test = train_model(X, y)
        evaluate_model(model, X_test, y_test)
    else:
        print("No data found.")
