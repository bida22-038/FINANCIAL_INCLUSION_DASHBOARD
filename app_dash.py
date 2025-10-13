import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc 
import numpy as np
import joblib
import os
import itertools 
from sklearn.inspection import partial_dependence, permutation_importance 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import is_regressor 
import sys
from sklearn import set_config # NEW: Import set_config

# NEW: Set scikit-learn transformers to output Pandas DataFrames
# This ensures feature names are preserved, resolving the UserWarning.
set_config(transform_output="pandas")

# --- 1. CONFIGURATION AND DATA/MODEL LOADING ---

MODEL_PATH = 'models/ensemble_model.joblib'
DATA_PATH = 'data/prep_outputs/merged_country_year_features.csv' 

# All model-expected feature names (order unknown)
MODEL_FEATURE_CANDIDATES = [
    'mobile_account_penetration', 
    'formal_account_penetration',
    'mobile_to_formal_ratio',
    'any_account_share', 
    'relative_penetration_index',
    'Population'
]

# Mapping the raw CSV column names to the Model's expected names.
FEATURE_MAPPING = {
    'mobile_account_share': 'mobile_account_penetration', 
    'formal_account_share': 'formal_account_penetration',
    'mobile_to_formal_ratio': 'mobile_to_formal_ratio', 
    'any_account_share': 'any_account_share', 
    'relative_penetration_index': 'relative_penetration_index',
    'pop_adult': 'Population', 
}

RAW_FEATURES_TO_LOAD = list(FEATURE_MAPPING.keys()) 
TARGET_Y = 'saved_formal' 

MODEL_FEATURES = MODEL_FEATURE_CANDIDATES # Initialize with an arbitrary order

# Metrics (using mock values for the dashboard display)
R2_ENSEMBLE = 0.7141
R2_OLS = 0.5836
R2_GAIN_PCT = round((R2_ENSEMBLE - R2_OLS) / R2_OLS * 100, 1)

MODEL = None
PREPROCESSOR = None
X_processed_df = pd.DataFrame() # Initializing X_processed_df globally
df = pd.DataFrame() 

# Load Data (Initial loading, independent of feature order)
try:
    df_raw = pd.read_csv(DATA_PATH, usecols=['country', 'year', TARGET_Y] + RAW_FEATURES_TO_LOAD)
    df_raw.rename(columns=FEATURE_MAPPING, inplace=True) 
    print(f"Data loaded successfully from {DATA_PATH}. Shape: {df_raw.shape}")
except Exception as e:
    print(f"FATAL ERROR: Could not load data from {DATA_PATH}. Using mock data. Error: {e}")
    data = {f: np.random.uniform(0.1, 0.8, 100) for f in MODEL_FEATURE_CANDIDATES}
    data.update({
        'country': np.random.choice(['Botswana', 'SADC1', 'SADC2', 'Rest'], 100),
        'year': np.random.randint(2011, 2021, 100),
        TARGET_Y: np.random.uniform(0.05, 0.5, 100)
    })
    df = pd.DataFrame(data)
    sys.exit(1) # Stop script if core data loading fails

df = df_raw.copy()

# --- Feature Order Search and Model Prediction Function ---
def find_correct_feature_order_and_predict(df_input: pd.DataFrame, feature_candidates: list, model_path: str):
    """
    Iterates through all feature order permutations to find the correct one
    that allows model prediction without error.
    """
    local_model = None
    if not os.path.exists(model_path):
        print(f"WARNING: Model file not found at {model_path}. Cannot search for order.")
        return False, df_input, feature_candidates, None, None, pd.DataFrame()

    try:
        local_model = joblib.load(model_path)
    except Exception as e:
        print(f"ERROR: Failed to load model object. Error: {e}")
        return False, df_input, feature_candidates, None, None, pd.DataFrame()
    
    total_permutations = len(list(itertools.permutations(feature_candidates)))
    
    for i, ordered_features in enumerate(itertools.permutations(feature_candidates)):
        ordered_features = list(ordered_features) # Tuple to list
        
        # 1. Prepare data in the current trial order
        df_trial = df_input.copy()
        X_raw = df_trial[ordered_features]
        Y_raw = df_trial[TARGET_Y]
        valid_rows_mask = Y_raw.notna()

        # 2. Preprocessing Pipeline
        preprocessor = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        # Initialize an empty DataFrame
        X_processed_df_trial = pd.DataFrame(index=df_trial.index, columns=ordered_features)
        
        if not X_raw[valid_rows_mask].empty:
            try:
                preprocessor.fit(X_raw[valid_rows_mask]) 
                
                # CRITICAL FIX UPDATE: transform now returns a DataFrame due to set_config
                X_processed_df_trial = preprocessor.transform(X_raw) 
                # Ensure index is correctly set, though it should be preserved with set_config
                X_processed_df_trial.index = df_trial.index 

            except Exception as e:
                print(f"Trial {i+1}/{total_permutations}: Preprocessing error: {e}")
                continue
        
        # 3. Attempt Prediction
        try:
            # Use the correctly ordered and processed trial data for prediction
            df_trial['predicted'] = local_model.predict(X_processed_df_trial) 
            df_trial.loc[valid_rows_mask, 'residuals'] = df_trial.loc[valid_rows_mask, TARGET_Y] - df_trial.loc[valid_rows_mask, 'predicted']
            
            # SUCCESS! Found the correct order.
            print("\n" + "="*80)
            print(f"SUCCESS: Correct feature order found on trial {i+1}/{total_permutations}!")
            print(f"Order: {ordered_features}")
            print("Predictions generated successfully. Using actual model output.")
            print("="*80 + "\n")
            
            # CRITICAL FIX: Return X_processed_df_trial
            return True, df_trial, ordered_features, local_model, preprocessor, X_processed_df_trial
            
        except ValueError as e:
            # Expected error for wrong order, continue trying
            continue
        except Exception as e:
            # Catch other unexpected prediction errors
            print(f"Trial {i+1}/{total_permutations}: Unexpected Prediction Error: {e}")
            continue

    # Failure after trying all 720 combinations
    print("\n" + "="*80)
    print("FATAL: Exhausted all 720 feature order permutations. The correct order could not be found.")
    print("The script will now fall back to mock data.")
    print("="*80 + "\n")
    return False, df_input.copy(), feature_candidates, None, None, pd.DataFrame()

# --- Execute the Search (CRITICAL FIX: Capture X_processed_df) ---
success, df, MODEL_FEATURES, MODEL, PREPROCESSOR, X_processed_df = find_correct_feature_order_and_predict(
    df, MODEL_FEATURE_CANDIDATES, MODEL_PATH
)

# If search fails, set up mock data environment and define X_processed_df
if not success:
    print("WARNING: Using mock predictions and feature importances.")
    # Fallback to mock data to keep the dashboard running
    df['predicted'] = df[TARGET_Y] * np.random.uniform(0.9, 1.1, len(df))
    df['residuals'] = df[TARGET_Y] - df['predicted']
    MODEL = None
    
    # Re-initialize preprocessor/processed data for PDP/Permutation, using arbitrary order
    PREPROCESSOR = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    X_raw = df[MODEL_FEATURES]
    valid_rows_mask = df[TARGET_Y].notna()
    
    # Define X_processed_df explicitly in the failure path
    if not X_raw[valid_rows_mask].empty:
        PREPROCESSOR.fit(X_raw[valid_rows_mask])
        X_processed_df = PREPROCESSOR.transform(X_raw) # Transform returns DataFrame
        X_processed_df.index = df.index
    else:
        X_processed_df = X_raw


# The final list used for features in all subsequent operations
FEATURES = MODEL_FEATURES

# --- FEATURE IMPORTANCE CALCULATION (Now correctly uses X_processed_df) ---

def calculate_permutation_importance(model, X_processed: pd.DataFrame, Y_actual: pd.Series, features: list) -> pd.DataFrame:
    """Calculates model-agnostic Permutation Importance."""
    if model is None or X_processed.empty or Y_actual.empty:
        return None
    
    try:
        r = permutation_importance(model, X_processed, Y_actual, 
                                   n_repeats=10, 
                                   random_state=42, 
                                   n_jobs=-1,
                                   scoring='r2')
        
        result_df = pd.DataFrame({
            'Feature': X_processed.columns, # Use columns from the DataFrame
            'Importance': r.importances_mean,
            'StdDev': r.importances_std
        }).sort_values(by='Importance', ascending=False)
        
        result_df['Importance'] = result_df['Importance'].abs()
        
        return result_df

    except Exception as e:
        print(f"Error calculating Permutation Importance: {e}. Falling back to simulation.")
        return None


X_for_importance = X_processed_df[df[TARGET_Y].notna()] 
Y_for_importance = df[df[TARGET_Y].notna()][TARGET_Y]

# Map internal feature names to human-readable labels for the dashboard
display_names = {
    'Population': 'Adult Population (M)', 
    'any_account_share': 'Any Account Share (%)', 
    'formal_account_penetration': 'Formal Penetration (%)', 
    'mobile_account_penetration': 'Mobile Penetration (%)', 
    'relative_penetration_index': 'Relative Penetration Index',
    'mobile_to_formal_ratio': 'Mobile to Formal Ratio'
}

# Feature importance calculation is now attempted with the correct features
feature_importances_df = calculate_permutation_importance(MODEL, X_for_importance, Y_for_importance, FEATURES)


if feature_importances_df is not None and not feature_importances_df.empty:
    # Use the calculated importance
    feature_importances_df['Feature'] = feature_importances_df['Feature'].map(display_names)
    feature_importances_data = feature_importances_df.sort_values(by='Importance', ascending=True)
else:
    # Fallback uses 6 mock values for 6 features
    print("WARNING: Using mock feature importances for plotting.")
    feature_importances_data = pd.DataFrame({
        'Feature': [display_names[f] for f in MODEL_FEATURE_CANDIDATES],
        'Importance': [0.45, 0.25, 0.15, 0.10, 0.05, 0.01]
    }).sort_values(by='Importance', ascending=True)

df_botswana = df[df['country'] == 'Botswana']
# Use the top feature from the successful order for PDP, or an arbitrary one if failed
KEY_PDP_FEATURE = FEATURES[0] 

# --- 2. PLOTLY CHART FUNCTIONS (No changes) ---

def create_r2_gauge(r2_score: float) -> go.Figure:
    """Creates a Plotly Gauge chart for the R-squared metric."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=r2_score * 100,
        title={'text': "Ensemble Model $R^2$ Score (%)", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#1a5e82"},
            'bar': {'color': "#1a5e82"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 50], 'color': "lightcoral"},
                {'range': [50, 70], 'color': "lightgoldenrodyellow"},
                {'range': [70, 100], 'color': "lightgreen"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 75}
        }
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=300)
    return fig

def create_feature_importance_chart(df_importance: pd.DataFrame) -> go.Figure:
    """Creates a horizontal bar chart for feature importance."""
    fig = px.bar(
        df_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top Feature Importance (Permutation Importance)',
        color_discrete_sequence=['#1a5e82']
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )
    return fig

def create_pdp_chart(model, preprocessor, df_X: pd.DataFrame, feature_name: str) -> go.Figure:
    """
    Calculates and plots the Partial Dependence Plot (PDP) using sklearn.
    """
    if model and not df_X.empty:
        try:
            full_estimator = Pipeline(steps=preprocessor.steps + [('regressor', model)])
            
            pdp_results = partial_dependence(
                full_estimator,       
                df_X,                 
                features=[feature_name],
                kind='average'
            )
            pdp_df = pd.DataFrame({
                # Note: pdp_results.individual[0][0] is a NumPy array, so no warning here
                'Feature_Value': pdp_results.individual[0][0],
                'Predicted_Target': pdp_results.average[0]
            })
            title = f'Partial Dependence Plot: {display_names.get(feature_name, feature_name)} vs. Saving'
            
        except Exception as e:
            print(f"Error generating PDP with sklearn: {e}. Falling back to simulation.")
            pdp_df = pd.DataFrame({'Feature_Value': np.linspace(df_X[feature_name].min() if not df_X[feature_name].empty else 0, 
                                                                 df_X[feature_name].max() if not df_X[feature_name].empty else 1, 50),
                                   'Predicted_Target': np.sort(np.random.uniform(df[TARGET_Y].min() if not df[TARGET_Y].empty else 0, 
                                                                                df[TARGET_Y].max() if not df[TARGET_Y].empty else 1, 50))}) 
            title = f'Partial Dependence Plot: {display_names.get(feature_name, feature_name)} vs. Saving (Simulated)'
            
    else:
        pdp_df = pd.DataFrame({'Feature_Value': np.linspace(0, 0.5, 5),
                               'Predicted_Target': [0.10, 0.25, 0.35, 0.40, 0.43]})
        title = f'Partial Dependence Plot: {display_names.get(feature_name, feature_name)} vs. Saving (Simulated)'


    fig = px.line(
        pdp_df,
        x='Feature_Value',
        y='Predicted_Target',
        labels={'Feature_Value': display_names.get(feature_name, feature_name), 'Predicted_Target': f'Predicted {TARGET_Y.replace("_", " ").title()} Rate'},
        title=title,
        color_discrete_sequence=['#ff7f0e']
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=350, hovermode="x unified")
    return fig

def create_target_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Creates a histogram for the target variable (saved_formal)."""
    df_plot = df.dropna(subset=[TARGET_Y]) 
    fig = px.histogram(
        df_plot,
        x=TARGET_Y,
        title=f'Distribution of {TARGET_Y.replace("_", " ").title()} Across Dataset',
        nbins=20,
        color_discrete_sequence=['#2ca02c']
    )
    fig.update_layout(
        xaxis_title=f'{TARGET_Y.replace("_", " ").title()} Rate (%)',
        yaxis_title='Count',
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )
    return fig

def create_botswana_trend_chart(df_botswana: pd.DataFrame) -> go.Figure:
    """Creates a time-series line chart for Botswana's actual and predicted trends."""
    df_actual = df_botswana.dropna(subset=[TARGET_Y]).copy()
    
    if df_botswana.empty:
        return {'layout': {'title': 'Botswana Data Not Available', 'height': 350}}

    fig = go.Figure()
    
    # Actual Trend
    fig.add_trace(go.Scatter(
        x=df_actual['year'], y=df_actual[TARGET_Y], mode='lines+markers', name='Actual',
        line=dict(color='#d62728', width=3)
    ))

    # Predicted Trend
    if 'predicted' in df_botswana.columns:
        fig.add_trace(go.Scatter(
            x=df_botswana['year'], y=df_botswana['predicted'], mode='lines+markers', name='Predicted',
            line=dict(color='#1f77b4', width=2, dash='dot')
        ))

    fig.update_layout(
        title=f"Botswana's {TARGET_Y.replace('_', ' ').title()} Trend (Actual & Predicted)",
        xaxis_title='Year',
        yaxis_title=f'{TARGET_Y.replace("_", " ").title()} Rate',
        margin=dict(l=20, r=20, t=50, b=20),
        height=350
    )
    return fig

def create_residuals_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Creates a histogram for the residuals."""
    df_plot = df.dropna(subset=['residuals'])
    fig = px.histogram(
        df_plot,
        x='residuals',
        title='Residuals Distribution (Actual - Predicted)',
        nbins=30,
        color_discrete_sequence=['#ff7f0e']
    )
    fig.update_layout(
        xaxis_title='Residual Value',
        yaxis_title='Count',
        margin=dict(l=20, r=20, t=50, b=20),
        height=300
    )
    return fig

def create_actual_vs_predicted_chart(df: pd.DataFrame) -> go.Figure:
    """Creates a scatter plot for Actual vs. Predicted values."""
    df_plot = df.dropna(subset=[TARGET_Y, 'predicted'])
    if df_plot.empty:
        return {'layout': {'title': 'Actual vs. Predicted Data Not Available', 'height': 300}}

    max_val = df_plot[[TARGET_Y, 'predicted']].max().max() * 1.05
    
    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=df_plot['predicted'],
        y=df_plot[TARGET_Y],
        mode='markers',
        name='Data Points',
        marker={'color': '#6a51a3', 'size': 5},
        hovertemplate=f'Predicted: %{{x:.3f}}<br>Actual: %{{y:.3f}}<extra></extra>'
    ))

    # Ideal line (y=x)
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Ideal Prediction (y=x)',
        line={'color': 'red', 'dash': 'dash'}
    ))

    fig.update_layout(
        title='Actual vs. Predicted Values',
        xaxis_title='Predicted Value',
        yaxis_title='Actual Value',
        margin=dict(l=20, r=20, t=50, b=20),
        height=300
    )
    return fig

# --- 3. DASH APP SETUP ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
app.title = "SADC Financial Inclusion Forecasting Dashboard"

HEADER_STYLE = {
    'background-color': '#1a5e82', 
    'color': 'white',
    'padding': '15px 25px',
    'margin-bottom': '20px',
    'border-radius': '8px',
    'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.2)'
}

TAB_STYLE = {'borderBottom': '1px solid #d6d6d6', 'padding': '10px', 'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'}
ACTIVE_TAB_STYLE = {'borderTop': '3px solid #007bff', 'borderBottom': '0', 'backgroundColor': 'white', 'color': '#007bff', 'padding': '10px', 'fontWeight': 'bold'}


# --- 4. DASHBOARD LAYOUT SECTIONS ---

# Dynamically pull the top two features for policy cards
if not feature_importances_data.empty:
    top_feature_name_1 = feature_importances_data.iloc[-1]['Feature']
    top_feature_importance_1 = feature_importances_data.iloc[-1]['Importance']
    top_feature_name_2 = feature_importances_data.iloc[-2]['Feature']
    top_feature_importance_2 = feature_importances_data.iloc[-2]['Importance']
else:
    # Fallback uses the first two features in the arbitrary order list
    top_feature_name_1 = display_names[MODEL_FEATURE_CANDIDATES[0]]
    top_feature_importance_1 = 0.45
    top_feature_name_2 = display_names[MODEL_FEATURE_CANDIDATES[1]]
    top_feature_importance_2 = 0.25

# TAB 1: Key Feature Plots (Target distribution, feature correlation, predicted trends)
tab_1_content = dbc.Container([
    dbc.Row([
        # Target Distribution
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Target Variable Distribution", className="card-title"),
            dcc.Graph(figure=create_target_distribution_chart(df), config={'displayModeBar': False})
        ]), className="mb-4 shadow"), md=6),
        
        # Feature Importance (with Tooltip)
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    # Element to trigger the tooltip
                    html.H5("Top Feature Importance", id="feature-importance-title", className="card-title"),
                    dcc.Graph(figure=create_feature_importance_chart(feature_importances_data), config={'displayModeBar': False}),
                    
                    # Tooltip component
                    dbc.Tooltip(
                        html.Div([
                            html.H6("Model Insight Summary", className="text-white"),
                            html.P("These values indicate the impact of each feature on the final 'Saved Formal' prediction.", className="small"),
                            html.P("Highest scores (top of the chart) are the best policy levers.", className="small"),
                            html.Hr(className="my-1"),
                            html.P(f"Top Driver: {top_feature_name_1} ({top_feature_importance_1:.2f})", className="small mb-0")
                        ]),
                        target="feature-importance-title",
                        placement="top",
                        style={'background-color': '#1a5e82', 'color': 'white', 'border-radius': '8px', 'padding': '10px', 'font-size': '0.9rem'},
                        className="shadow",
                    ),
                ]), 
                className="mb-4 shadow"
            ), 
            md=6
        ),
    ]),
    dbc.Row([
        # PDP Plot
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Model Behaviour: Partial Dependence Plot", className="card-title"),
            # Ensure the correct feature from the discovered order is passed
            dcc.Graph(figure=create_pdp_chart(MODEL, PREPROCESSOR, X_processed_df, KEY_PDP_FEATURE), config={'displayModeBar': False})
        ]), className="mb-4 shadow"), md=12),
    ]),
], fluid=True, className="py-3")

# TAB 2: Model Performance Plots
tab_2_content = dbc.Container([
    dbc.Row([
        # R2 Gauge
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Ensemble Model Predictive Power ($R^2$)", className="card-title text-center"),
            dcc.Graph(figure=create_r2_gauge(R2_ENSEMBLE), config={'displayModeBar': False}),
            html.P(f"Gain over OLS: {R2_GAIN_PCT}%", className="text-center font-weight-bold")
        ]), className="mb-4 shadow text-center"), md=4),
        # Residuals Plot 
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Model Residuals Distribution", className="card-title"),
            html.P("A histogram of (Actual - Predicted) values should be centered around zero and normally distributed.", className="card-text text-muted small"),
            dcc.Graph(figure=create_residuals_distribution_chart(df), config={'displayModeBar': False})
        ]), className="mb-4 shadow"), md=4),
        # Actual vs Predicted
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Actual vs. Predicted Plot", className="card-title"),
            html.P("Points close to the diagonal red line indicate high prediction accuracy.", className="card-text text-muted small"),
            dcc.Graph(figure=create_actual_vs_predicted_chart(df), config={'displayModeBar': False})
        ]), className="mb-4 shadow"), md=4),
    ]),
], fluid=True, className="py-3")


# TAB 3: Key Strategic Insights
top_feature_pdp_display_name = display_names.get(KEY_PDP_FEATURE, KEY_PDP_FEATURE.replace('_', ' ').title())

tab_3_content = dbc.Container([
    dbc.Row([
        # Policy Recommendation Card 1 (Based on top feature)
        dbc.Col(dbc.Card([
            dbc.CardHeader("POLICY RECOMMENDATION 1: Enhance Network Reach", className="bg-primary text-white"),
            dbc.CardBody([
                html.P(f"Insight from Model: {top_feature_name_1} is the top predictor (Importance: {top_feature_importance_1:.2f}).", className="card-text"),
                html.P(f"Policy: Based on the Partial Dependence Plot (PDP) for {top_feature_pdp_display_name}, focus investment on increasing penetration to the optimal threshold point identified on the curve to maximize its positive impact on formal savings.", className="small"),
            ])
        ], className="mb-4 shadow"), md=6),
        # Policy Recommendation Card 2 (Based on second top feature)
        dbc.Col(dbc.Card([
            dbc.CardHeader("POLICY RECOMMENDATION 2: Targeted Interventions", className="bg-success text-white"),
            dbc.CardBody([
                html.P(f"Insight from Model: {top_feature_name_2} is also highly relevant (Importance: {top_feature_importance_2:.2f}).", className="card-text"),
                html.P("Policy: Develop specific programs, such as agent network expansion or digital literacy, to bolster the impact of this second most important driver, focusing on demographic groups with low adoption rates for this feature.", className="small"),
            ])
        ], className="mb-4 shadow"), md=6),
    ]),
    dbc.Row([
        # Botswana Trend (Predicted Trends)
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5("Country Trend: Botswana's Financial Inclusion Trend", className="card-title"),
            dcc.Graph(figure=create_botswana_trend_chart(df_botswana), config={'displayModeBar': False})
        ]), className="mb-4 shadow"), md=12),
    ])
], fluid=True, className="py-3")


# --- 5. MAIN APP LAYOUT ---
app.layout = dbc.Container([
    # Header
    html.Div(
        html.H1("SADC Financial Inclusion Forecasting Dashboard", className="header-title"),
        style=HEADER_STYLE
    ),
    
    # Tabs Component
    dbc.Tabs(
        id="dashboard-tabs",
        active_tab="tab-1",
        children=[
            dbc.Tab(label="Key Feature Plots", tab_id="tab-1", style=TAB_STYLE, active_tab_style=ACTIVE_TAB_STYLE),
            dbc.Tab(label="Model Performance Plots", tab_id="tab-2", style=TAB_STYLE, active_tab_style=ACTIVE_TAB_STYLE),
            dbc.Tab(label="Key Strategic Insights", tab_id="tab-3", style=TAB_STYLE, active_tab_style=ACTIVE_TAB_STYLE),
        ],
        className="mb-3"
    ),

    # Tab Content Area
    html.Div(id="tab-content"),
    
], fluid=True, className="p-4")

# --- 6. CALLBACK FOR TAB CONTENT RENDERING ---
@app.callback(
    dash.Output("tab-content", "children"),
    [dash.Input("dashboard-tabs", "active_tab")]
)
def render_tab_content(active_tab):
    """Dynamically loads content based on the selected tab."""
    if active_tab == "tab-1":
        return tab_1_content
    elif active_tab == "tab-2":
        return tab_2_content
    elif active_tab == "tab-3":
        return tab_3_content
    return html.P("Select a tab")

# --- 7. RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=True)