import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import config

class Explainer:
    """
    Generates SHAP explanations for a model prediction.
    """
    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train
        self.feature_names = X_train.columns
        self.shap_explainer = None
        self.preprocessor = None

    def fit_explainer(self):
        """
        Creates a SHAP explainer based on the trained model.
        """
        print("\nInitializing SHAP explainer...")
        
        # Extract the preprocessor steps (imputer + scaler) from the pipeline
        self.preprocessor = self.model[0:2]
        
        # Extract the classifier model
        classifier = self.model.named_steps['clf']
        
        # Transform the training data to be used as the background for the explainer
        X_train_transformed = self.preprocessor.transform(self.X_train)
        
        self.shap_explainer = shap.Explainer(classifier, X_train_transformed)
        print("SHAP explainer fitted.")

    def explain_sample(self, sample_X):
        """
        Calculates SHAP values for a single sample and generates a waterfall plot.
        """
        if not self.shap_explainer:
            print("❌ Error: SHAP explainer not fitted. Call fit_explainer() first.")
            return None, None
        
        print("Explaining prediction for the sample...")
   
        # Transform the single sample using the same preprocessor
        sample_X_transformed = self.preprocessor.transform(sample_X)

        # Compute SHAP values using the unified API
        shap_values = self.shap_explainer(sample_X_transformed)

        # Generate and save the waterfall plot for the first sample
        plot_path = os.path.join(config.REPORTS_DIR, config.SHAP_PLOT_FILENAME)

        exp = shap.Explanation(
            values=shap_values[0].values if hasattr(shap_values[0], 'values') else shap_values[0],
            base_values=getattr(shap_values, 'base_values', 0),
            data=sample_X_transformed[0],
            feature_names=self.feature_names
        )

        # Now plot with feature names shown
        shap.plots.waterfall(exp, show=False)

        # Save the figure
        fig = plt.gcf()
        fig.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

        print(f"SHAP waterfall plot saved to: {plot_path}")

        # Step 6: Get top 5 important features (by absolute SHAP value)
        shap_array = shap_values.values[0]  # array of SHAP values for this sample
        feature_names = self.feature_names
        abs_shap = np.abs(shap_array)
        top5_idx = np.argsort(abs_shap)[-5:][::-1]  # indices of top 5 features

        # Step 7: Print or extract top 5 SHAP values
        top5_features = feature_names[top5_idx].values
        top5_shap_values = shap_array[top5_idx]
        top5_features_values = sample_X.iloc[0, top5_idx].values

        features_zip = list(zip(top5_features, top5_shap_values, top5_features_values))
        # # Display results
        # for feature, shap_value, vals in features_zip:
        #     print(f"{feature}: value = {vals:.4f} SHAP value = {shap_value:.4f}")

        features_formatted = []
        for feature, shap_value, vals in features_zip:
            print(f"- {feature}: {vals:.4f} (SHAP : {shap_value:.4f})")
            features_formatted.append(f"- {feature}: {vals:.4f} (SHAP : {shap_value:.4f})")
            "\n".join(features_formatted)

        return features_formatted, plot_path