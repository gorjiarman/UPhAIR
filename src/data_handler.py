import pandas as pd
from sklearn.model_selection import train_test_split
import os
import config
import SimpleITK as sitk
from radiomics import featureextractor
import numpy as np

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
    

    def load_sample(self):
        """
        Extract PyRadiomics features from NIfTI image + mask
        and return them as:
            - features_array
            - feature_names
            - first_slice_image
            - middle_slice_image
            - last_slice_image
        """

        # Default radiomics parameters
        default_params = {
            "binWidth": 25,
            "label": 1,
            "interpolator": sitk.sitkBSpline,
            "correctMask": True
        }

        # Initialize PyRadiomics extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()
        for k, v in default_params.items():
            extractor.settings[k] = v

        extractor.enableImageTypeByName("Original")
        extractor.enableAllFeatures()

        # Load image + mask
        image = sitk.ReadImage("data/sample/UCSF-PDGM-0540_T1.nii.gz")
        mask = sitk.ReadImage("data/sample/UCSF-PDGM-0540_tumor_segmentation.nii.gz")
        mask[mask == 2] = 1

        # ----------------------------
        # Radiomics extraction
        # ----------------------------
        result = extractor.execute(image, mask)

        feature_names = []
        feature_values = []

        for k, v in result.items():
            if k.startswith("diagnostics_"):
                continue
            try:
                val = float(v)
                feature_names.append(k)
                feature_values.append(val)
            except:
                pass

        features_array = np.array(feature_values, dtype=float)

        # ----------------------------
        # Slice extraction + contour overlay
        # ----------------------------

        # Convert to NumPy
        img_np = sitk.GetArrayFromImage(image)      # shape: [z, y, x]
        mask_np = sitk.GetArrayFromImage(mask)
        mask_np[mask_np == 2] = 1


        z = img_np.shape[0]

        # --- Find slices where segmentation exists ---
        mask_sums = mask_np.reshape(z, -1).sum(axis=1)       # sum of each slice
        seg_slices = np.where(mask_sums > 0)[0]              # indices with tumor

        if len(seg_slices) == 0:
            # No segmentation found, fallback to full volume definition
            first_idx = 0
            mid_idx = z // 2
            last_idx = z - 1
        else:
            # First, middle, and last slices containing segmentation
            first_idx = seg_slices[0]
            last_idx  = seg_slices[-1]
            mid_idx   = seg_slices[len(seg_slices) // 2]

        slice_indices = [first_idx, mid_idx, last_idx]
        rendered_slices = []

        for idx in slice_indices:
            img_slice = img_np[idx]
            mask_slice = mask_np[idx]

            # Create an RGB image
            rgb = np.stack([img_slice, img_slice, img_slice], axis=-1)
            rgb = rgb - rgb.min()
            if rgb.max() > 0:
                rgb = rgb / rgb.max()
            rgb = (rgb * 255).astype(np.uint8)

            # Build contour mask
            from scipy.ndimage import binary_dilation

            border = mask_slice.astype(bool) ^ binary_dilation(mask_slice.astype(bool))
            rgb[border] = [255, 0, 0]  # red contour

            rendered_slices.append(rgb)

        first_slice_img, middle_slice_img, last_slice_img = rendered_slices
        clincal_fs = pd.read_csv("data/sample/UCSF-PDGM-0540-clinical.csv")
        
        clinical_arr = np.array(clincal_fs.loc[0, :].values.tolist())
        features_array = np.append(clinical_arr,features_array)

        clinical_arr_n = np.array(clincal_fs.columns.values.tolist())
        feature_names = np.append(clinical_arr_n,feature_names)

        return (
            features_array,
            feature_names,
            first_slice_img,
            middle_slice_img,
            last_slice_img
        )
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