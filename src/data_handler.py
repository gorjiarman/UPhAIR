import pandas as pd
from sklearn.model_selection import train_test_split
import os
import config

class DataHandler:
    """
    Handles loading, splitting, and preparing the dataset.
    """
    def __init__(self, csv_filename=config.DATASET_PATH):
        self.filepath = os.path.join(config.DATA_DIR, csv_filename)
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.sample_X = None
        self.sample_y = None
        self.sample_info = {}

    def load_data(self):
        """Loads the dataset from the CSV file."""
        print(f"Loading data from {self.filepath}...")
        try:
            self.df = pd.read_csv(self.filepath)
            print("Data loaded successfully.")
            return True
        except FileNotFoundError:
            print(f"❌ Error: The file was not found at {self.filepath}")
            print("Please ensure your dataset is in the 'data/' directory.")
            return False

    def prepare_and_split_data(self):
        """Prepares and splits the data for training and a single sample prediction."""
        if self.df is None:
            print("❌ Error: Data not loaded. Call load_data() first.")
            return

        print("Preparing data for training and sampling...")
        
        # Isolate the data for training (all except the sample)
        training_df = self.df.drop(self.df.index[config.SAMPLE_INDEX])
        
        # Isolate the single sample for prediction
        sample_df = self.df.iloc[[config.SAMPLE_INDEX]]

        # Prepare training data
        self.y = training_df[config.LABEL_COLUMN]
        self.X = training_df.drop(columns=[config.PATIENT_ID_COLUMN, config.LABEL_COLUMN])

        # Prepare sample data
        self.sample_y = sample_df[config.LABEL_COLUMN].iloc[0]
        self.sample_X = sample_df.drop(columns=[config.PATIENT_ID_COLUMN, config.LABEL_COLUMN])
        
        # Store other sample info for the report
        self.sample_info['PatientID'] = sample_df[config.PATIENT_ID_COLUMN].iloc[0]
        if 'age' in sample_df.columns:
            self.sample_info['Age'] = sample_df['age'].iloc[0]
        else:
            self.sample_info['Age'] = "N/A"
        
        # Split the main dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE,
            stratify=self.y # Stratify for balanced splits in classification
        )
        print("Data preparation complete.")
        print(f"  Training set size: {self.X_train.shape[0]} samples")
        print(f"  Test set size: {self.X_test.shape[0]} samples")
        print(f"  Single sample prepared for prediction (Patient ID: {self.sample_info['PatientID']}).")