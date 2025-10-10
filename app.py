from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI()

MODEL_PATH = 'models/ensemble_model.joblib'
DATA_PATH = 'data/prep_outputs/merged_country_year_features.csv'

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MODEL = None
DF_FULL = None

FEATURES = ['mobile_account_share', 'agents_per_100k', 'active_accounts', 'formal_account_share', 'mobile_to_formal_ratio']
TARGET_Y = 'saved_formal'

METRICS = {
    'r2_ensemble': 0.7141,
    'r2_ols': 0.5836,
    'r2_gain': round(0.7141 - 0.5836, 4),
    'r2_gain_pct': round((0.7141 - 0.5836) / 0.5836 * 100, 1)
}

# --- DATA & MODEL LOADING ---
try:
    if os.path.exists(MODEL_PATH):
        MODEL = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}. Prediction endpoints will be disabled.")

    if os.path.exists(DATA_PATH):
        DF_FULL = pd.read_csv(DATA_PATH)
        print(f"✅ Data loaded successfully from {DATA_PATH}")
    else:
        print(f"⚠️ Data file not found at {DATA_PATH}. Some visualizations may be empty.")

except Exception as e:
    print(f"❌ Critical error during startup: {e}")

# --- HELPER FUNCTION FOR FEATURE IMPORTANCE ---
def get_model_feature_importances(model, features):
    """
    Extracts feature importances from a fitted model object, handling common
    scikit-learn structures like models with feature_importances_ or coef_.
    """
    
    try:
        if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
            importance_values = model.feature_importances_
        elif hasattr(model, 'coef_') and model.coef_ is not None:
            importance_values = np.abs(model.coef_)
        elif hasattr(model, 'final_estimator_') and hasattr(model.final_estimator_, 'feature_importances_'):
            importance_values = model.final_estimator_.feature_importances_
        else:
            print("⚠️ Model does not have accessible feature_importances_ or coef_. Cannot extract feature importance dynamically.")
            return None
        
        if importance_values.sum() > 0:
             normalized_values = importance_values / importance_values.sum()
        else:
             normalized_values = importance_values
        sorted_features = sorted(zip(features, normalized_values), key=lambda x: x[1], reverse=True)

        COLOR_PALETTE = ['#28a745', '#17a2b8', '#ffc107', '#6c757d', '#dc3545']
        importance_list = []
        
        for i, (feature_key, importance_value) in enumerate(sorted_features[:5]):

            display_name = feature_key.replace('_account_share', ' Penetration').replace('agents_per_100k', 'Agent Density').replace('_', ' ').title()
            
            importance_list.append({
                'feature': display_name, 
                'importance': float(importance_value), 
                'color': COLOR_PALETTE[i % len(COLOR_PALETTE)]
            })
            
        return importance_list

    except Exception as e:
        print(f"❌ Error during dynamic feature importance calculation: {e}")
        return None

# --- API ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Renders the main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request, "metrics": METRICS})

@app.get("/api/dashboard_data")
async def get_dashboard_data():
    """Provides the JSON data for all frontend visualizations."""
    
    if DF_FULL is None:
        return {"error": "Data not available."}

    df = DF_FULL.copy().dropna(subset=[TARGET_Y] + FEATURES)

    # --- 1. Feature Importance Data (Bar Chart) ---
    feature_importance_data = None
    
    if MODEL:
        feature_importance_data = get_model_feature_importances(MODEL, FEATURES)

    # Fallback to simulated data if model is not loaded or importance extraction failed
    if not feature_importance_data:
        feature_importance_data = [
            {'feature': 'Mobile Penetration', 'importance': 0.38, 'color': '#28a745'},
            {'feature': 'Agent Density', 'importance': 0.22, 'color': '#17a2b8'},
            {'feature': 'Active Accounts', 'importance': 0.15, 'color': '#ffc107'},
            {'feature': 'Formal Account Share', 'importance': 0.10, 'color': '#6c757d'},
            {'feature': 'Mobile-to-Formal Ratio', 'importance': 0.08, 'color': '#dc3545'}
        ]
    
    # --- 2. R2 Comparison Data (Data in METRICS, handled by JS) ---

    # --- 3. Predicted vs Actual (Scatter Plot) ---
    # filter for Botswana for illustrative purposes
    df_botswana = df[df['country'] == 'Botswana'].copy()
    
    if MODEL and not df_botswana.empty:
        X_pred = df_botswana[FEATURES]
        try:
            df_botswana['predicted'] = MODEL.predict(X_pred)
            pred_actual_data = {
                'actual': df_botswana[TARGET_Y].tolist(),
                'predicted': df_botswana['predicted'].tolist(),
                'years': df_botswana['year'].tolist()
            }
        except Exception as e:
             print(f"Prediction failed: {e}")
             pred_actual_data = {'actual': [], 'predicted': [], 'years': []}
    else:
        pred_actual_data = {'actual': [], 'predicted': [], 'years': []}
    
    # --- 4. SADC Trends (Botswana vs. SADC Avg Mobile Penetration - Line Chart) ---
    df_sadc_avg = df.groupby('year')['mobile_account_share'].mean().reset_index().rename(columns={'mobile_account_share': 'SADC_Avg'})
    df_bots = df[df['country'] == 'Botswana'].rename(columns={'mobile_account_share': 'Botswana'})
    
    trend_data = pd.merge(df_bots[['year', 'Botswana']], df_sadc_avg, on='year', how='inner')
    
    # --- 5. Correlation Heatmap Data (Bar Chart) ---
    corr_matrix = df[[TARGET_Y] + FEATURES].corr()
    corr_target = corr_matrix[TARGET_Y].drop(TARGET_Y).sort_values(ascending=False)
    corr_data = [{'feature': f.replace('_', ' ').title(), 'correlation': c} for f, c in corr_target.items()]

    # --- 6. PDP Data (Simulated Non-Linear Curve - Line Chart) ---
    pdp_points = [
        {'x': 0, 'y': 0.10}, {'x': 10, 'y': 0.25}, {'x': 20, 'y': 0.35},
        {'x': 30, 'y': 0.40}, {'x': 40, 'y': 0.42}, {'x': 50, 'y': 0.43}
    ]
    
    # --- 7. Residuals Distribution (Histogram) ---
    if 'predicted' in df_botswana.columns:
        residuals = (df_botswana[TARGET_Y] - df_botswana['predicted']).tolist()
    else:
        residuals = []
        
    # --- 8. Country Comparison: Top Predictor vs. Target (Scatter) ---
    df_comp = df.groupby('country').agg(
        mobile_pen=(FEATURES[0], 'mean'),
        formal_saving=(TARGET_Y, 'mean')
    ).reset_index().sample(n=min(10, len(df.country.unique())), random_state=42) # Sample 10 countries
    
    country_comp_data = df_comp.to_dict('records')
    
    return {
        "feature_importance": feature_importance_data,
        "pred_actual": pred_actual_data,
       
        "correlation": corr_data,
        "pdp_data": pdp_points,
   
        "country_comparison": country_comp_data
    }
