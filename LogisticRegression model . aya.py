"""
Machine Learning model for dark store location prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import pickle

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. ML features will be disabled.")

from .config import Config


logger = logging.getLogger(__name__)


class MLModel:
    """Machine Learning model for location suitability prediction."""
    
    def __init__(self, config: Config):
        """Initialize ML model."""
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.model_metrics = {}
        self.model_type = config.get('ml.model_type', 'random_forest')
        self.models = {}  # Store multiple models
        self.all_metrics = {}  # Store metrics for all models
    
    def train_model(self, df: pd.DataFrame, target_column: str = 'final_score') -> Dict:
        """Train ML model to predict location suitability."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, skipping ML training")
            return {}
        
        logger.info(f"Training {self.model_type} model")
        
        # Select features
        feature_cols = self._select_feature_columns(df, target_column)
        
        if not feature_cols:
            raise ValueError("No suitable features found for ML training")
        
        # Prepare data
        X = df[feature_cols].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        X = X.fillna(0)
        y = y.fillna(0)
        
        # Remove non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        logger.info(f"Training with {len(X)} samples and {len(X.columns)} features")
        
        # Split data
        test_size = self.config.get('ml.test_size', 0.2)
        random_state = self.config.get('ml.random_state', 42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=self.config.get('ml.n_estimators', 100),
                max_depth=self.config.get('ml.max_depth', 10),
                random_state=random_state
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=self.config.get('ml.n_estimators', 100),
                max_depth=self.config.get('ml.max_depth', 10),
                random_state=random_state
            )
        else:
            logger.warning(f"Unknown model type {self.model_type}, using Random Forest")
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=random_state
            )
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        self.model_metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'feature_names': feature_cols
        }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=self.config.get('ml.cv_folds', 5),
            scoring='r2'
        )
        self.model_metrics['cv_mean_r2'] = cv_scores.mean()
        self.model_metrics['cv_std_r2'] = cv_scores.std()
        
        logger.info(f"Model training completed. Test R²: {self.model_metrics['test_r2']:.4f}")
        
        return self.model_metrics
    
    def train_multiple_models(self, df: pd.DataFrame, target_column: str = 'final_score') -> Dict:
        """Train multiple ML models and compare results."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, skipping ML training")
            return {}
        
        logger.info("Training multiple ML models for comparison")
        
        # Select features
        feature_cols = self._select_feature_columns(df, target_column)
        
        if not feature_cols:
            raise ValueError("No suitable features found for ML training")
        
        # Prepare data for regression
        X = df[feature_cols].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        X = X.fillna(0)
        y = y.fillna(0)
        
        # Remove non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        logger.info(f"Training with {len(X)} samples and {len(X.columns)} features")
        
        # Split data
        test_size = self.config.get('ml.test_size', 0.2)
        random_state = self.config.get('ml.random_state', 42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define regression models to train
        model_configs = {
            'random_forest': RandomForestRegressor(
                n_estimators=self.config.get('ml.n_estimators', 100),
                max_depth=self.config.get('ml.max_depth', 10),
                random_state=random_state
            ),
            'linear_regression': LinearRegression()
        }
        
        # Train regression models
        for model_name, model in model_configs.items():
            logger.info(f"Training {model_name}")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = {
                'train_mse': mean_squared_error(y_train, y_train_pred),
                'test_mse': mean_squared_error(y_test, y_test_pred),
                'train_r2': r2_score(y_train, y_train_pred),
                'test_r2': r2_score(y_test, y_test_pred),
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'test_mae': mean_absolute_error(y_test, y_test_pred),
                'feature_names': feature_cols
            }
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train,
                cv=self.config.get('ml.cv_folds', 5),
                scoring='r2'
            )
            metrics['cv_mean_r2'] = cv_scores.mean()
            metrics['cv_std_r2'] = cv_scores.std()
            
            # Store model and metrics
            self.models[model_name] = model
            self.all_metrics[model_name] = metrics
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            logger.info(f"{model_name} completed. Test R²: {metrics['test_r2']:.4f}")
        
        # Train Logistic Regression for classification
        logger.info("Training logistic_regression for classification")
        
        # Convert continuous scores to discrete classes
        y_class = self._convert_to_classes(y)
        y_train_class = self._convert_to_classes(y_train)
        y_test_class = self._convert_to_classes(y_test)
        
        # Train Logistic Regression
        log_reg = LogisticRegression(random_state=random_state, max_iter=1000)
        log_reg.fit(X_train_scaled, y_train_class)
        
        # Make predictions
        y_train_pred_class = log_reg.predict(X_train_scaled)
        y_test_pred_class = log_reg.predict(X_test_scaled)
        
        # Calculate classification metrics
        class_metrics = {
            'train_accuracy': accuracy_score(y_train_class, y_train_pred_class),
            'test_accuracy': accuracy_score(y_test_class, y_test_pred_class),
            'feature_names': feature_cols,
            'num_classes': len(np.unique(y_class)),
            'classes': list(np.unique(y_class))
        }
        
        # Cross-validation for classification
        cv_scores = cross_val_score(
            log_reg, X_train_scaled, y_train_class,
            cv=self.config.get('ml.cv_folds', 5),
            scoring='accuracy'
        )
        class_metrics['cv_mean_accuracy'] = cv_scores.mean()
        class_metrics['cv_std_accuracy'] = cv_scores.std()
        
        # Store model and metrics
        self.models['logistic_regression'] = log_reg
        self.all_metrics['logistic_regression'] = class_metrics
        
        logger.info(f"logistic_regression completed. Test Accuracy: {class_metrics['test_accuracy']:.4f}")
        
        # Set the best regression model as the main model
        regression_models = {k: v for k, v in self.all_metrics.items() if k != 'logistic_regression'}
        best_model_name = max(regression_models.keys(), 
                            key=lambda k: regression_models[k]['test_r2'])
        self.model = self.models[best_model_name]
        self.model_metrics = self.all_metrics[best_model_name]
        self.model_type = best_model_name
        
        logger.info(f"Best regression model: {best_model_name} with Test R²: {self.model_metrics['test_r2']:.4f}")
        
        return self.all_metrics
    
    def _convert_to_classes(self, y: pd.Series, num_classes: int = 3) -> np.ndarray:
        """Convert continuous values to discrete classes."""
        # Use quantiles to create balanced classes
        return pd.qcut(y, q=num_classes, labels=False, duplicates='drop')
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Select features
        feature_cols = self.model_metrics['feature_names']
        X = df[feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Remove non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def _select_feature_columns(self, df: pd.DataFrame, target_column: str) -> list:
        """Select feature columns for ML training."""
        # Exclude target and non-predictive columns
        exclude_cols = [
            target_column, 'rank', 'selected', 'name', 'category',
            'latitude', 'longitude', 'poi_categories', 'area_type'
        ]
        
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out excluded columns
        feature_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Filter out columns with too many missing values
        missing_threshold = self.config.get('ml.missing_threshold', 0.5)
        feature_cols = [
            col for col in feature_cols 
            if df[col].isna().sum() / len(df) < missing_threshold
        ]
        
        return feature_cols
    
    def save_model(self, output_dir: Path) -> None:
        """Save trained model and scaler."""
        if self.model is None:
            logger.warning("No model to save")
            return
        
        logger.info(f"Saving model to {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all models if multiple were trained
        if self.models:
            for model_name, model in self.models.items():
                model_path = output_dir / f"{model_name}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Saved {model_name} to {model_path}")
        else:
            # Save single model
            model_path = output_dir / "dark_store_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Saved model to {model_path}")
        
        # Save scaler
        scaler_path = output_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Save feature importance
        if self.feature_importance is not None:
            importance_path = output_dir / "feature_importance.csv"
            self.feature_importance.to_csv(importance_path, index=False)
            logger.info(f"Saved feature importance to {importance_path}")
        
        # Save metrics (all models if multiple)
        if self.all_metrics:
            metrics_df = pd.DataFrame(self.all_metrics).T
            metrics_path = output_dir / "all_model_metrics.csv"
            metrics_df.to_csv(metrics_path)
            logger.info(f"Saved all model metrics to {metrics_path}")
        else:
            metrics_path = output_dir / "model_metrics.csv"
            metrics_df = pd.DataFrame([self.model_metrics])
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Saved model metrics to {metrics_path}")
    
    def load_model(self, model_path: Path, scaler_path: Path) -> None:
        """Load trained model and scaler."""
        logger.info(f"Loading model from {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info("Model and scaler loaded successfully")
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of model performance."""
        if not self.model_metrics:
            return pd.DataFrame()
        
        summary = {
            'Model Type': self.model_type,
            'Train R²': f"{self.model_metrics.get('train_r2', 0):.4f}",
            'Test R²': f"{self.model_metrics.get('test_r2', 0):.4f}",
            'Train MSE': f"{self.model_metrics.get('train_mse', 0):.4f}",
            'Test MSE': f"{self.model_metrics.get('test_mse', 0):.4f}",
            'Train MAE': f"{self.model_metrics.get('train_mae', 0):.4f}",
            'Test MAE': f"{self.model_metrics.get('test_mae', 0):.4f}",
            'CV Mean R²': f"{self.model_metrics.get('cv_mean_r2', 0):.4f}",
            'CV Std R²': f"{self.model_metrics.get('cv_std_r2', 0):.4f}",
            'Num Features': len(self.model_metrics.get('feature_names', []))
        }
        
        return pd.DataFrame([summary])
