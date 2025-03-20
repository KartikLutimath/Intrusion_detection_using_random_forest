import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

def load_dataset(file_path, col_names):
    """
    Load the NSL-KDD dataset from a given file path.
    The NSL-KDD files do not have header rows, so we provide column names.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    return pd.read_csv(file_path, header=None, names=col_names)

def perform_eda(df):
    """
    Perform basic exploratory data analysis (EDA):
    - Print statistical summary.
    - Plot histograms for numerical features.
    """
    print("Statistical Summary:")
    print(df.describe())

    # Plot histograms for numerical features (limit the number for clarity)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols].hist(figsize=(12, 10))
    plt.tight_layout()
    # Uncomment the next line if you want to view the plots
    # plt.show()

def main():
    # Define column names commonly used for NSL-KDD
    col_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
        "logged_in", "num_compromised", "root_shell", "su_attempted",
        "num_root", "num_file_creations", "num_shells", "num_access_files",
        "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
        "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
        "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "label"
    ]

    # Set paths to the training and testing datasets
    base_dir = os.path.join(os.getcwd(), "data")
    train_file = os.path.join(base_dir, "KDDTrain+.txt")
    test_file = os.path.join(base_dir, "KDDTest+.txt")

    # Load datasets
    print("Loading training dataset...")
    train_df = load_dataset(train_file, col_names)
    print("Loading testing dataset...")
    test_df = load_dataset(test_file, col_names)

    # Combine train and test for overall preprocessing
    df = pd.concat([train_df, test_df], ignore_index=True)
    print("Combined dataset shape:", df.shape)

    # Perform Exploratory Data Analysis (EDA)
    print("\nPerforming Exploratory Data Analysis...")
    perform_eda(df)

    # Drop any rows with missing values (if they exist)
    df = df.dropna()

    # Check which columns are still of object type
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    print("Object-type columns before encoding:", object_cols)

    # List of known categorical columns to be label-encoded.
    categorical_cols = [
        "protocol_type", "service", "flag",
        "land", "logged_in", "is_host_login", "is_guest_login"
    ]

    # Label-encode each categorical column if it exists in the DataFrame.
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # Check again for any remaining object-type columns.
    remaining_object_cols = df.select_dtypes(include=['object']).columns.tolist()
    print("Object-type columns after encoding:", remaining_object_cols)

    # Convert any remaining non-label columns to numeric.
    # We'll keep the "label" column as is.
    for col in df.columns:
        if col != "label" and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Verify conversion
    final_object_cols = df.select_dtypes(include=['object']).columns.tolist()
    print("Remaining object-type columns after conversion:", final_object_cols)

    # Split features and labels
    X = df.drop("label", axis=1)
    y = df["label"]

    # Normalize and scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Data Splitting: 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Save the preprocessed data to CSV files
    output_dir = os.path.join(os.getcwd(), "preprocessed_data")
    os.makedirs(output_dir, exist_ok=True)

    preprocessed_train = pd.DataFrame(X_train, columns=X.columns)
    preprocessed_train["label"] = y_train.values
    preprocessed_train.to_csv(os.path.join(output_dir, "train_preprocessed.csv"), index=False)

    preprocessed_test = pd.DataFrame(X_test, columns=X.columns)
    preprocessed_test["label"] = y_test.values
    preprocessed_test.to_csv(os.path.join(output_dir, "test_preprocessed.csv"), index=False)

    print("\nPreprocessing complete. Preprocessed files saved in:", output_dir)

if __name__ == "__main__":
    main()
