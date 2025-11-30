# New Building Prediction Strategy
**Handling Out-of-Sample Buildings in NYC Vacancy Risk Model**

## Problem Statement
Users may enter BBL numbers or addresses for buildings not in our training dataset of 7,191 office buildings. We need a robust strategy to handle these cases gracefully while maintaining prediction quality.

## Solution Architecture

### **1. Prediction Pipeline for New Buildings**

```python
import pandas as pd
import joblib
from src.data_loader import get_pluto_data
from src.feature_engineering import extract_clean_features

class NewBuildingPredictor:
    def __init__(self, model_path, scaler_path, feature_columns):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = feature_columns
        self.pluto_data = get_pluto_data()  # Full NYC PLUTO dataset
        
    def predict_new_building(self, bbl_or_address):
        """Generate prediction for building not in training set."""
        
        # 1. Look up building in PLUTO
        building_data = self._lookup_building(bbl_or_address)
        if building_data is None:
            return {'error': 'Building not found in NYC records'}
            
        # 2. Verify it's an office building
        if not self._is_office_building(building_data):
            return {'error': 'Not an office building - model focuses on office properties'}
            
        # 3. Extract same features used in training
        try:
            features = extract_clean_features(building_data)
            features_df = pd.DataFrame([features], columns=self.feature_columns)
            
            # 4. Apply same preprocessing
            features_scaled = self.scaler.transform(features_df)
            
            # 5. Generate prediction
            probability = self.model.predict_proba(features_scaled)[0, 1]
            
            return {
                'bbl': building_data['bbl'],
                'address': building_data.get('address', 'Address not available'),
                'probability': float(probability),
                'risk_level': self._categorize_risk(probability),
                'confidence': 'Medium - New Building Prediction',
                'note': 'Prediction based on building characteristics, not historical patterns'
            }
            
        except Exception as e:
            return {'error': f'Could not generate prediction: {str(e)}'}
    
    def _lookup_building(self, bbl_or_address):
        """Look up building in PLUTO dataset."""
        if bbl_or_address.replace('-', '').replace(' ', '').isdigit():
            # BBL lookup
            bbl_clean = bbl_or_address.replace('-', '').replace(' ', '')
            matches = self.pluto_data[self.pluto_data['bbl'] == bbl_clean]
        else:
            # Address lookup (simplified)
            matches = self.pluto_data[
                self.pluto_data['address'].str.contains(bbl_or_address, case=False, na=False)
            ]
        
        return matches.iloc[0] if len(matches) > 0 else None
    
    def _is_office_building(self, building_data):
        """Check if building qualifies as office building."""
        office_classes = ['O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9']
        return (
            building_data.get('bldgclass', '')[:2] in office_classes or
            building_data.get('officearea', 0) > 0 or
            building_data.get('comarea', 0) > building_data.get('resarea', 0)
        )
    
    def _categorize_risk(self, probability):
        """Convert probability to risk category."""
        if probability >= 0.7:
            return 'High Risk'
        elif probability >= 0.3:
            return 'Medium Risk'
        else:
            return 'Low Risk'
```

### **2. Dashboard Integration**

```python
# In your Streamlit dashboard
@st.cache_data
def load_existing_predictions():
    """Load pre-computed predictions for 7,191 buildings."""
    return pd.read_csv('results/building_predictions.csv')

@st.cache_resource  
def load_new_building_predictor():
    """Initialize predictor for new buildings."""
    return NewBuildingPredictor(
        model_path='models/baseline_model.pkl',
        scaler_path='models/scaler.pkl', 
        feature_columns=['building_age', 'office_ratio', ...]  # Your 20 features
    )

def smart_building_lookup(user_input):
    """Handle any building lookup with fallback strategy."""
    
    # Load resources
    existing_preds = load_existing_predictions()
    new_predictor = load_new_building_predictor()
    
    # 1. Check existing predictions first (fast)
    existing_match = existing_preds[
        (existing_preds['bbl'] == user_input) | 
        (existing_preds['address'].str.contains(user_input, case=False, na=False))
    ]
    
    if len(existing_match) > 0:
        result = existing_match.iloc[0]
        return {
            'source': 'Training Dataset',
            'bbl': result['bbl'],
            'address': result['address'],
            'probability': result['probability'],
            'risk_level': result['risk_level'],
            'confidence': 'High - From Training Data'
        }
    
    # 2. Try new building prediction
    st.info("🔍 Building not in training dataset - generating new prediction...")
    return new_predictor.predict_new_building(user_input)

# Usage in Streamlit
def building_lookup_page():
    st.title("🏢 Building Risk Lookup")
    
    user_input = st.text_input("Enter BBL or Address:")
    
    if user_input:
        with st.spinner("Looking up building..."):
            result = smart_building_lookup(user_input)
            
        if 'error' in result:
            st.error(result['error'])
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Vacancy Probability", f"{result['probability']:.1%}")
            with col2:  
                st.metric("Risk Level", result['risk_level'])
            with col3:
                st.metric("Confidence", result['confidence'])
                
            st.info(f"📍 **Address:** {result['address']}")
            st.info(f"🏗️ **BBL:** {result['bbl']}")
            
            if 'note' in result:
                st.warning(f"⚠️ **Note:** {result['note']}")
```

### **3. Data Expansion Strategy**

```python
# Pre-compute predictions for ALL NYC office buildings
def expand_prediction_database():
    """Expand from 7,191 to all NYC office buildings."""
    
    # Load full PLUTO data
    pluto_full = pd.read_csv('data/raw/pluto_25v2_1.csv')
    
    # Identify all office buildings (broader than training set)
    office_filter = (
        (pluto_full['bldgclass'].str[:2].isin(['O1', 'O2', 'O3', 'O4', 'O5'])) |
        (pluto_full['officearea'] > 1000) |  # Significant office space
        ((pluto_full['comarea'] > pluto_full['resarea']) & (pluto_full['comarea'] > 0))
    )
    
    all_office_buildings = pluto_full[office_filter].copy()
    print(f"Found {len(all_office_buildings)} potential office buildings")
    
    # Generate predictions for all
    predictor = NewBuildingPredictor(...)
    
    predictions = []
    for _, building in all_office_buildings.iterrows():
        try:
            pred = predictor.predict_single_building(building)
            predictions.append(pred)
        except Exception as e:
            print(f"Failed to predict for {building['bbl']}: {e}")
    
    # Save expanded predictions
    expanded_df = pd.DataFrame(predictions)
    expanded_df.to_csv('results/all_office_building_predictions.csv', index=False)
    print(f"Generated predictions for {len(expanded_df)} buildings")
```

## User Experience Strategy

### **Dashboard Flow:**
1. **Fast Path**: Check pre-computed predictions (7,191 buildings)
2. **Real-time Path**: Generate new prediction if office building
3. **Graceful Failure**: Clear error messages for non-office or invalid input

### **User Feedback:**
- ✅ **Found in dataset**: "High confidence prediction from training data"
- 🔄 **New office building**: "Generated prediction - medium confidence" 
- ❌ **Not office building**: "Our model focuses on office properties"
- 🚫 **Invalid input**: "Please check BBL format or address spelling"

This approach ensures your dashboard can handle any building lookup gracefully while maintaining prediction quality!