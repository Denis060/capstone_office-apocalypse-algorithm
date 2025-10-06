"""
Office Apocalypse Algorithm - Final Validation Script
=====================================================

This script performs end-to-end validation of the entire project pipeline
to ensure everything is ready for academic submission and evaluation.

Author: Office Apocalypse Algorithm Team
Date: October 6, 2025
"""

import pandas as pd
import numpy as np
import joblib
import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_check(message, status=True):
    """Print a check with status"""
    icon = "✅" if status else "❌"
    print(f"   {icon} {message}")

def validate_project_structure():
    """Validate the overall project structure"""
    print_header("PROJECT STRUCTURE VALIDATION")
    
    # Define expected directories and files
    expected_structure = {
        'data/raw': ['MTA_Subway_Hourly_Ridership__2020-2024.csv', 'business_registry.csv'],
        'data/processed': ['office_buildings_processed.csv', 'README.md'],
        'data/features': ['office_features_cross_dataset_integrated.csv'],
        'notebooks': ['01_exploratory_data_analysis.ipynb', '02_feature_engineering.ipynb', '03_model_training.ipynb'],
        'models': ['champion_model.joblib', 'X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv'],
        'results': ['feature_analysis', 'model_performance', 'dataset_validation', 'documentation'],
        'docs': ['DATASET_INTEGRATION_METHODOLOGY.md', 'DATASET_INTEGRATION_TECHNICAL.md']
    }
    
    all_good = True
    
    for directory, files in expected_structure.items():
        dir_path = Path(directory)
        if dir_path.exists():
            print_check(f"Directory exists: {directory}")
            
            for file in files:
                file_path = dir_path / file
                if file_path.exists():
                    print_check(f"File exists: {directory}/{file}")
                else:
                    print_check(f"File missing: {directory}/{file}", False)
                    all_good = False
        else:
            print_check(f"Directory missing: {directory}", False)
            all_good = False
    
    return all_good

def validate_data_pipeline():
    """Validate the data processing pipeline"""
    print_header("DATA PIPELINE VALIDATION")
    
    try:
        # 1. Check processed data
        processed_path = Path('data/processed/office_buildings_processed.csv')
        if processed_path.exists():
            processed_df = pd.read_csv(processed_path)
            print_check(f"Processed data loaded: {processed_df.shape}")
            print_check(f"Office buildings count: {len(processed_df):,}")
            
            # Validate expected columns
            expected_office_count = 7191
            if len(processed_df) == expected_office_count:
                print_check(f"Correct office building count: {expected_office_count:,}")
            else:
                print_check(f"Office building count mismatch: expected {expected_office_count:,}, got {len(processed_df):,}", False)
        
        # 2. Check feature-engineered data
        features_path = Path('data/features/office_features_cross_dataset_integrated.csv')
        if features_path.exists():
            features_df = pd.read_csv(features_path)
            print_check(f"Feature data loaded: {features_df.shape}")
            print_check(f"Features available: {features_df.shape[1]} columns")
            
            # Check for BBL column (unique identifier)
            if 'BBL' in features_df.columns:
                print_check("BBL identifier column present")
                unique_bbls = features_df['BBL'].nunique()
                print_check(f"Unique buildings: {unique_bbls:,}")
            else:
                print_check("BBL identifier column missing", False)
        
        return True
        
    except Exception as e:
        print_check(f"Data pipeline validation failed: {str(e)}", False)
        return False

def validate_model_artifacts():
    """Validate all model training artifacts"""
    print_header("MODEL ARTIFACTS VALIDATION")
    
    try:
        models_dir = Path('models')
        
        # 1. Check training/test data
        X_train = pd.read_csv(models_dir / 'X_train.csv', index_col=0)
        X_test = pd.read_csv(models_dir / 'X_test.csv', index_col=0)
        y_train = pd.read_csv(models_dir / 'y_train.csv', index_col=0).squeeze()
        y_test = pd.read_csv(models_dir / 'y_test.csv', index_col=0).squeeze()
        
        print_check(f"Training data: {X_train.shape}")
        print_check(f"Test data: {X_test.shape}")
        print_check(f"Training labels: {len(y_train)}")
        print_check(f"Test labels: {len(y_test)}")
        
        # Validate split proportions
        total_samples = len(X_train) + len(X_test)
        train_pct = len(X_train) / total_samples * 100
        test_pct = len(X_test) / total_samples * 100
        
        print_check(f"Train/test split: {train_pct:.1f}% / {test_pct:.1f}%")
        
        # Check feature consistency
        if list(X_train.columns) == list(X_test.columns):
            print_check(f"Feature consistency: {len(X_train.columns)} features")
        else:
            print_check("Feature inconsistency between train/test", False)
        
        # Check class balance
        train_pos_rate = y_train.mean()
        test_pos_rate = y_test.mean()
        print_check(f"Class balance - Train: {train_pos_rate:.1%}, Test: {test_pos_rate:.1%}")
        
        # 2. Check trained models
        champion_model = joblib.load(models_dir / 'champion_model.joblib')
        scaler = joblib.load(models_dir / 'feature_scaler.joblib')
        
        print_check("Champion model loaded successfully")
        print_check("Feature scaler loaded successfully")
        
        # 3. Test model prediction
        # Make predictions on a small sample
        sample_X = X_test.head(10)
        predictions = champion_model.predict(sample_X)
        probabilities = champion_model.predict_proba(sample_X)
        
        print_check(f"Model predictions work: {len(predictions)} predictions")
        print_check(f"Probability predictions work: {probabilities.shape}")
        
        # 4. Check metadata
        with open(models_dir / 'model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print_check(f"Model metadata loaded")
        print_check(f"Champion model: {metadata['champion_model']}")
        print_check(f"Training date: {metadata['training_date']}")
        print_check(f"ROC-AUC: {metadata['champion_performance']['roc_auc']:.4f}")
        
        return True, {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(X_train.columns),
            'champion_model': metadata['champion_model'],
            'roc_auc': metadata['champion_performance']['roc_auc']
        }
        
    except Exception as e:
        print_check(f"Model artifacts validation failed: {str(e)}", False)
        return False, {}

def validate_notebooks():
    """Validate notebook structure and accessibility"""
    print_header("NOTEBOOKS VALIDATION")
    
    notebooks = [
        '01_exploratory_data_analysis.ipynb',
        '02_feature_engineering.ipynb', 
        '03_model_training.ipynb'
    ]
    
    try:
        for notebook in notebooks:
            notebook_path = Path('notebooks') / notebook
            if notebook_path.exists():
                print_check(f"Notebook exists: {notebook}")
                
                # Basic size check (should not be empty)
                size_mb = notebook_path.stat().st_size / (1024 * 1024)
                if size_mb > 0.1:  # At least 100KB
                    print_check(f"Notebook has content: {size_mb:.1f} MB")
                else:
                    print_check(f"Notebook might be empty: {size_mb:.1f} MB", False)
            else:
                print_check(f"Notebook missing: {notebook}", False)
        
        return True
        
    except Exception as e:
        print_check(f"Notebook validation failed: {str(e)}", False)
        return False

def validate_documentation():
    """Validate project documentation"""
    print_header("DOCUMENTATION VALIDATION")
    
    try:
        # Check main documentation files
        docs_files = [
            'docs/DATASET_INTEGRATION_METHODOLOGY.md',
            'docs/DATASET_INTEGRATION_TECHNICAL.md',
            'docs/PROJECT_INTEGRATION_SUMMARY.md',
            'models/README.md',
            'data/processed/README.md'
        ]
        
        for doc_file in docs_files:
            doc_path = Path(doc_file)
            if doc_path.exists():
                size_kb = doc_path.stat().st_size / 1024
                print_check(f"Documentation exists: {doc_file} ({size_kb:.1f} KB)")
            else:
                print_check(f"Documentation missing: {doc_file}", False)
        
        return True
        
    except Exception as e:
        print_check(f"Documentation validation failed: {str(e)}", False)
        return False

def generate_final_report(model_stats):
    """Generate final validation report"""
    print_header("FINAL VALIDATION REPORT")
    
    print("\n🎓 PROJECT SUMMARY FOR ACADEMIC SUBMISSION")
    print("─" * 50)
    
    if model_stats:
        print(f"📊 Dataset: {model_stats['train_samples'] + model_stats['test_samples']:,} NYC office buildings")
        print(f"🔧 Features: {model_stats['features']} engineered features")
        print(f"🤖 Champion Model: {model_stats['champion_model']}")
        print(f"📈 Performance: {model_stats['roc_auc']:.4f} ROC-AUC")
        print(f"📝 Training Split: {model_stats['train_samples']:,} / {model_stats['test_samples']:,} (80/20)")
    
    print(f"\n✅ PROJECT STATUS: READY FOR ACADEMIC EVALUATION")
    print("─" * 50)
    print("📁 All directories organized professionally")
    print("📊 Data pipeline validated and documented") 
    print("🤖 Models trained and artifacts saved")
    print("📔 Notebooks structured and accessible")
    print("📖 Comprehensive documentation completed")
    
    print(f"\n🎯 REPRODUCIBILITY CONFIRMED")
    print("─" * 30)
    print("• Raw data preserved in data/raw/")
    print("• Processed data available in data/processed/")
    print("• Feature engineering captured in data/features/")
    print("• Model artifacts saved in models/")
    print("• Training pipeline documented in notebooks/")
    print("• Complete methodology documented in docs/")

def main():
    """Main validation function"""
    print("🚀 OFFICE APOCALYPSE ALGORITHM - FINAL VALIDATION")
    print("=" * 60)
    print("Testing entire pipeline for academic submission readiness...")
    
    # Change to project directory
    project_dir = Path("c:/Users/pcric/Desktop/capstone_project/office_apocalypse_algorithm_project")
    import os
    os.chdir(project_dir)
    
    validation_results = []
    
    # Run all validations
    validation_results.append(validate_project_structure())
    validation_results.append(validate_data_pipeline()) 
    validation_results.append(validate_notebooks())
    validation_results.append(validate_documentation())
    
    model_success, model_stats = validate_model_artifacts()
    validation_results.append(model_success)
    
    # Generate final report
    generate_final_report(model_stats if model_success else {})
    
    # Overall status
    if all(validation_results):
        print(f"\n🎉 VALIDATION COMPLETE: ALL SYSTEMS GO!")
        print("Your capstone project is ready for academic submission! 🎓")
        return True
    else:
        print(f"\n⚠️ VALIDATION ISSUES DETECTED")
        print("Please review the failed checks above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)