"""
Temporal Validation Strategy for Office Apocalypse Algorithm

This module implements time-aware validation strategies to prevent data leakage
and ensure robust model evaluation for vacancy prediction.

KEY PRINCIPLES:
==============
1. TEMPORAL SPLIT: Train on historical data, validate on future data
2. NO DATA LEAKAGE: Features from future cannot inform past predictions
3. REALISTIC EVALUATION: Mimics real-world deployment scenario

VALIDATION STRATEGIES:
====================
1. Simple Temporal Split (70/15/15)
2. Rolling Window Cross-Validation  
3. Expanding Window Cross-Validation
4. Geographic Stratified Temporal Split

IMPLEMENTATION:
==============
- Handles time-series aspects of building data
- Maintains chronological order
- Supports multiple validation approaches
- Provides comprehensive evaluation metrics
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class TemporalValidator:
    """
    Implements temporal validation strategies for time-series aware model evaluation.
    
    Prevents data leakage by ensuring training data is always from earlier time periods
    than validation/test data.
    """
    
    def __init__(self, data: pd.DataFrame, target_col: str = 'is_vacant', 
                 time_col: str = 'data_date'):
        """
        Initialize temporal validator with dataset.
        
        Args:
            data: DataFrame with features and target
            target_col: Name of target variable column
            time_col: Name of temporal column for splitting
        """
        self.data = data.copy()
        self.target_col = target_col
        self.time_col = time_col
        
        # Ensure temporal column exists or create one
        if time_col not in data.columns:
            print(f"Warning: {time_col} not found. Creating synthetic temporal column.")
            self._create_synthetic_temporal_column()
    
    def _create_synthetic_temporal_column(self):
        """Create synthetic temporal column based on available data patterns."""
        # Use index as proxy for time if no temporal column exists
        n_records = len(self.data)
        start_date = datetime(2020, 1, 1)  # Start from 2020 for realistic timeline
        
        # Create quarterly time stamps
        dates = []
        for i in range(n_records):
            quarter_offset = i % 16  # 4 years * 4 quarters
            date_offset = timedelta(days=quarter_offset * 90)  # Quarterly
            dates.append(start_date + date_offset)
        
        self.data[self.time_col] = dates
        print(f"Created synthetic temporal column: {min(dates)} to {max(dates)}")
    
    def temporal_split(self, train_size: float = 0.7, val_size: float = 0.15, 
                      test_size: float = 0.15) -> Tuple[pd.DataFrame, ...]:
        """
        Perform temporal split maintaining chronological order.
        
        Args:
            train_size: Proportion for training (earliest data)
            val_size: Proportion for validation (middle data)
            test_size: Proportion for testing (latest data)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs((train_size + val_size + test_size) - 1.0) < 1e-10, \
            "Split proportions must sum to 1.0"
        
        # Sort by temporal column
        sorted_data = self.data.sort_values(self.time_col).reset_index(drop=True)
        
        n_total = len(sorted_data)
        train_end = int(n_total * train_size)
        val_end = int(n_total * (train_size + val_size))
        
        train_df = sorted_data.iloc[:train_end].copy()
        val_df = sorted_data.iloc[train_end:val_end].copy()
        test_df = sorted_data.iloc[val_end:].copy()
        
        # Print split statistics
        self._print_split_stats(train_df, val_df, test_df)
        
        return train_df, val_df, test_df
    
    def _print_split_stats(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                          test_df: pd.DataFrame):
        """Print statistics about the temporal split."""
        print("\\nTEMPORAL SPLIT STATISTICS:")
        print("=" * 50)
        
        for name, df in [("TRAIN", train_df), ("VALIDATION", val_df), ("TEST", test_df)]:
            time_range = f"{df[self.time_col].min()} to {df[self.time_col].max()}"
            class_dist = df[self.target_col].value_counts(normalize=True)
            
            print(f"\\n{name} SET:")
            print(f"  Size: {len(df):,} records ({len(df)/len(self.data)*100:.1f}%)")
            print(f"  Time Range: {time_range}")
            print(f"  Class Distribution: {dict(class_dist.round(3))}")
    
    def rolling_window_validation(self, window_size: int = 1000, 
                                 min_train_size: int = 2000,
                                 n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Implement rolling window cross-validation for time series.
        
        Args:
            window_size: Size of validation window
            min_train_size: Minimum size for training set
            n_splits: Number of splits to create
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        sorted_data = self.data.sort_values(self.time_col)
        n_total = len(sorted_data)
        
        splits = []
        
        # Calculate step size between splits
        step_size = max(1, (n_total - min_train_size - window_size) // (n_splits - 1))
        
        for i in range(n_splits):
            train_end = min_train_size + i * step_size
            val_start = train_end
            val_end = min(val_start + window_size, n_total)
            
            if val_end - val_start < window_size // 2:  # Ensure minimum validation size
                break
                
            train_indices = np.arange(0, train_end)
            val_indices = np.arange(val_start, val_end)
            
            splits.append((train_indices, val_indices))
        
        print(f"\\nROLLING WINDOW VALIDATION: Created {len(splits)} splits")
        return splits
    
    def expanding_window_validation(self, min_train_size: int = 2000,
                                  val_window_size: int = 500,
                                  n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Implement expanding window validation (train set grows over time).
        
        Args:
            min_train_size: Initial training set size
            val_window_size: Validation window size
            n_splits: Number of splits
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        sorted_data = self.data.sort_values(self.time_col)
        n_total = len(sorted_data)
        
        splits = []
        
        # Calculate step size
        remaining_data = n_total - min_train_size - val_window_size
        step_size = max(1, remaining_data // (n_splits - 1))
        
        for i in range(n_splits):
            train_end = min_train_size + i * step_size
            val_start = train_end
            val_end = min(val_start + val_window_size, n_total)
            
            if val_end - val_start < val_window_size // 2:
                break
                
            train_indices = np.arange(0, train_end)
            val_indices = np.arange(val_start, val_end)
            
            splits.append((train_indices, val_indices))
        
        print(f"\\nEXPANDING WINDOW VALIDATION: Created {len(splits)} splits")
        return splits
    
    def geographic_stratified_split(self, geographic_col: str = 'borough', 
                                   train_size: float = 0.7) -> Tuple[pd.DataFrame, ...]:
        """
        Perform temporal split while maintaining geographic distribution.
        
        Args:
            geographic_col: Column containing geographic information
            train_size: Proportion for training
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if geographic_col not in self.data.columns:
            print(f"Warning: {geographic_col} not found. Using regular temporal split.")
            return self.temporal_split(train_size)
        
        # Perform stratified temporal split by geographic region
        sorted_data = self.data.sort_values(self.time_col)
        
        train_dfs, val_dfs, test_dfs = [], [], []
        
        for geo_region in sorted_data[geographic_col].unique():
            geo_data = sorted_data[sorted_data[geographic_col] == geo_region].copy()
            
            if len(geo_data) < 10:  # Skip regions with too little data
                continue
            
            # Temporal split within geographic region
            n_total = len(geo_data)
            train_end = int(n_total * train_size)
            val_end = int(n_total * (train_size + 0.15))
            
            train_dfs.append(geo_data.iloc[:train_end])
            val_dfs.append(geo_data.iloc[train_end:val_end])
            test_dfs.append(geo_data.iloc[val_end:])
        
        # Combine all geographic regions
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        print(f"\\nGEOGRAPHIC STRATIFIED SPLIT:")
        print(f"Maintained distribution across {len(sorted_data[geographic_col].unique())} regions")
        self._print_split_stats(train_df, val_df, test_df)
        
        return train_df, val_df, test_df
    
    def evaluate_validation_strategy(self, model, X_train: pd.DataFrame, 
                                   y_train: pd.Series, X_val: pd.DataFrame, 
                                   y_val: pd.Series, strategy_name: str = ""):
        """
        Evaluate a model using the validation strategy.
        
        Args:
            model: Trained model with predict and predict_proba methods
            X_train, y_train: Training data
            X_val, y_val: Validation data
            strategy_name: Name of validation strategy for reporting
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'strategy': strategy_name,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_val, y_pred, average='weighted', zero_division=0),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_val, y_pred_proba)
        
        return metrics
    
    def plot_temporal_distribution(self, save_path: str = None):
        """Plot temporal distribution of data and target variable."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Data distribution over time
        self.data[self.time_col].hist(bins=50, ax=ax1, alpha=0.7, color='skyblue')
        ax1.set_title('Data Distribution Over Time')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Number of Records')
        
        # Plot 2: Target variable distribution over time
        monthly_avg = self.data.groupby(self.data[self.time_col].dt.to_period('M'))[self.target_col].mean()
        monthly_avg.plot(ax=ax2, color='orange', linewidth=2)
        ax2.set_title('Vacancy Rate Over Time')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Vacancy Rate')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Temporal distribution plot saved to: {save_path}")
        
        plt.show()

def demonstrate_validation_strategies():
    """Demonstrate different temporal validation strategies."""
    
    print("TEMPORAL VALIDATION STRATEGY DEMONSTRATION")
    print("=" * 60)
    
    # Load processed data
    data_path = Path("data/processed/office_buildings_processed.csv")
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure you have run the feature engineering pipeline first.")
        return
    
    # Load data
    data = pd.read_csv(data_path)
    print(f"Loaded data: {len(data):,} records with {len(data.columns)} features")
    
    # Create target variable if it doesn't exist
    if 'is_vacant' not in data.columns:
        # Use the existing target from the data
        if 'target_high_vacancy_risk' in data.columns:
            data['is_vacant'] = data['target_high_vacancy_risk']
        else:
            # Create synthetic target based on risk factors
            data['is_vacant'] = (
                (data['vacancy_risk_alert'] == 'Orange') | 
                (data['vacancy_risk_alert'] == 'Red')
            ).astype(int) if 'vacancy_risk_alert' in data.columns else np.random.choice([0, 1], len(data), p=[0.85, 0.15])
        
        print(f"Created target variable 'is_vacant' - Vacancy rate: {data['is_vacant'].mean():.3f}")
    
    # Initialize validator
    validator = TemporalValidator(data)
    
    # Strategy 1: Simple Temporal Split
    print("\\n1. SIMPLE TEMPORAL SPLIT")
    print("-" * 30)
    train_df, val_df, test_df = validator.temporal_split()
    
    # Strategy 2: Rolling Window Cross-Validation
    print("\\n2. ROLLING WINDOW CROSS-VALIDATION")
    print("-" * 40)
    rolling_splits = validator.rolling_window_validation(n_splits=3)
    
    # Strategy 3: Expanding Window Cross-Validation  
    print("\\n3. EXPANDING WINDOW CROSS-VALIDATION")
    print("-" * 42)
    expanding_splits = validator.expanding_window_validation(n_splits=3)
    
    # Strategy 4: Geographic Stratified Split
    print("\\n4. GEOGRAPHIC STRATIFIED TEMPORAL SPLIT")
    print("-" * 45)
    # Check if borough column exists, if not create synthetic one
    if 'borough' not in data.columns:
        boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
        data['borough'] = np.random.choice(boroughs, len(data))
    
    geo_train, geo_val, geo_test = validator.geographic_stratified_split()
    
    # Plot temporal distributions
    print("\\n5. TEMPORAL DISTRIBUTION ANALYSIS")
    print("-" * 40)
    validator.plot_temporal_distribution()
    
    return validator, (train_df, val_df, test_df)

if __name__ == "__main__":
    # Demonstrate validation strategies
    validator, splits = demonstrate_validation_strategies()
    
    print("\\nTEMPORAL VALIDATION SETUP COMPLETE!")
    print("=" * 50)
    print("Next steps:")
    print("1. Use temporal_split() for baseline model evaluation")
    print("2. Apply rolling_window_validation() for robust cross-validation") 
    print("3. Implement geographic_stratified_split() for spatial generalization")
    print("4. Proceed to Task 4.2: Build Baseline Model")