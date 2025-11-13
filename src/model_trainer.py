import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import config

class ModelTrainer:
    """
    Trains and evaluates multiple classifiers to find the best one.
    """
    def __init__(self, classifiers_dict):
        self.classifiers = classifiers_dict
        self.best_model = None
        self.best_classifier_name = None
        self.results_df = None

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Loops through classifiers, trains them using GridSearchCV, and evaluates performance.
        """
        results = []
        print("\n--- Starting Model Training and Evaluation ---")
        
        for name, (clf, params) in self.classifiers.items():
            print(f"\nTraining {name}...")
            
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('clf', clf)
            ])
            
            # Use GridSearchCV to find the best hyperparameters
            grid = GridSearchCV(pipeline, param_grid=params, cv=5, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            
            model = grid.best_estimator_
            
            # Predictions and Probabilities
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Store results
            test_auc = roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else np.nan
            results.append({
                'Classifier': name,
                'Test Accuracy': accuracy_score(y_test, y_test_pred),
                'Test F1-Score': f1_score(y_test, y_test_pred, average='weighted'),
                'Test AUC': test_auc,
                'Best Params': grid.best_params_
            })
            
            print(f"  ✅ {name} finished with Test AUC: {test_auc:.4f}")

        self.results_df = pd.DataFrame(results).sort_values(by='Test AUC', ascending=False)
        
        # Identify and store the best model
        best_classifier_info = self.results_df.iloc[0]
        self.best_classifier_name = best_classifier_info['Classifier']
        
        best_clf, best_params = self.classifiers[self.best_classifier_name]
        
        # Re-train the best model on the full training data with the best params
        print(f"\nRe-training the best model ({self.best_classifier_name}) on the full training set...")
        self.best_model = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', best_clf.set_params(**{k.replace('clf__', ''): v for k, v in best_classifier_info['Best Params'].items()}))
        ])
        self.best_model.fit(X_train, y_train)
        
        print("--- Model Training Complete ---")
        print(f"🏆 Best performing model: {self.best_classifier_name}")
        print("\nPerformance Results:")
        print(self.results_df[['Classifier', 'Test AUC', 'Test Accuracy', 'Test F1-Score']].to_string(index=False))

    def predict_sample(self, sample_X):
        """
        Makes a prediction on a single sample using the best trained model.
        """
        if not self.best_model:
            print("❌ Error: No model has been trained. Run train_and_evaluate() first.")
            return None, None
        
        prediction = self.best_model.predict(sample_X)[0]
        prediction_proba = self.best_model.predict_proba(sample_X)[0]
        
        predicted_class = "IDH-wildtype" if prediction == 1 else "IDH-mutant"
        print(f"\nPrediction for sample: {predicted_class} (Raw: {prediction})")
        print(f"Prediction Probabilities: {prediction_proba}")
        
        return prediction, predicted_class