document.addEventListener('DOMContentLoaded', () => {
    fetchDashboardData();
        // Select all cards that should have the hover insight
    document.querySelectorAll('.chart-card').forEach(card => {
        const insight = card.querySelector('.hover-insight');
        const closeButton = card.querySelector('.close-insight');

        if (insight && closeButton) {
            let dismissed = false; // Flag to track if the user clicked 'X'

            // 1. Show insight on mouse enter
            card.addEventListener('mouseenter', () => {
                if (!dismissed) {
                    insight.classList.add('insight-visible');
                }
            });

            // 2. Hide insight on mouse leave
            card.addEventListener('mouseleave', () => {
                // Small delay to prevent flicker if the cursor grazes the edge
                setTimeout(() => {
                   if (!card.matches(':hover')) { 
                        insight.classList.remove('insight-visible');
                        // Reset dismissed flag so hover works next time
                        if (!dismissed) {
                            dismissed = false; 
                        }
                   }
                }, 50); 
            });

            // 3. Explicitly hide and set dismissed flag on 'X' click
            closeButton.addEventListener('click', (e) => {
                e.preventDefault(); 
                e.stopPropagation(); // Stop click from affecting the card beneath
                dismissed = true; 
                insight.classList.remove('insight-visible');
            });
        }
    });
});

// Fetches data from the FastAPI backend
async function fetchDashboardData() {
    try {
        const response = await fetch('/api/dashboard_data');
        const data = await response.json();
        
        if (data.error) {
            console.error(data.error);
            document.getElementById('chart-importance').innerHTML = `<p class="text-danger text-center mt-5">Error: ${data.error}</p>`;
            return;
        }

        // Draw all 8 visualizations
        drawFeatureImportance(data.feature_importance); // 1. Bar Chart
        drawPDP(data.pdp_data);                         // 2. Line Chart
        drawR2Comparison();                             // 3. Bar Chart (Data from Jinja context)
        drawPredVsActual(data.pred_actual);             // 4. Scatter Plot
    
        drawCorrelation(data.correlation);              // 6. Bar Chart
   
        drawCountryComparison(data.country_comparison); // 8. Scatter Plot

        // Gauge uses hardcoded metrics from Jinja template
        const r2_score = parseFloat(document.querySelector('.metric-card.bg-primary .card-title').textContent) * 100;
        drawGauge(r2_score);

    } catch (error) {
        console.error("Error fetching dashboard data:", error);
    }
}

// --- CHART DRAWING FUNCTIONS (Using Plotly.js) ---

// 1. Feature Importance (Bar Chart)
function drawFeatureImportance(data) {
    const features = data.map(d => d.feature).reverse();
    const importance = data.map(d => d.importance).reverse();
    const colors = data.map(d => d.color).reverse();

    const trace = {
        x: importance,
        y: features,
        type: 'bar',
        orientation: 'h',
        marker: { color: colors }
    };

    const layout = {
        title: { text: 'Impact Score on Saving Rate' },
        margin: { l: 150, r: 20, t: 50, b: 50 },
        xaxis: { title: 'Normalized Importance', range: [0, 0.4] },
        height: 350
    };
    Plotly.newPlot('chart-importance', [trace], layout, {responsive: true, displayModeBar: false});
}

// 2. Partial Dependence Plot (Line Chart)
function drawPDP(data) {
    const x = data.map(d => d.x);
    const y = data.map(d => d.y);

    const trace = {
        x: x, y: y, mode: 'lines+markers', type: 'scatter',
        name: 'Partial Dependence',
        line: { color: '#dc3545', width: 4, shape: 'spline' }
    };

    const layout = {
        title: { text: 'Saving Rate vs. Mobile Penetration' },
        xaxis: { title: 'Mobile Penetration (%)' },
        yaxis: { title: 'Predicted Formal Saving Rate' },
        height: 350
    };
    Plotly.newPlot('chart-pdp', [trace], layout, {responsive: true, displayModeBar: false});
}

// 3. R2 Comparison (Bar Chart)
function drawR2Comparison() {
    // Get R2 values from the metric cards using DOM access
    const r2_ensemble = parseFloat(document.querySelector('.metric-card.bg-primary .card-title').textContent);
    const r2_ols = 0.5836; // Hardcoded OLS baseline

    const models = ['Ensemble ML', 'OLS Baseline'];
    const r2_values = [r2_ensemble, r2_ols];

    const trace = {
        x: models,
        y: r2_values,
        type: 'bar',
        marker: { color: ['#007bff', '#ffc107'] }
    };

    const layout = {
        title: { text: 'Model R² Comparison (CV)' },
        yaxis: { title: 'R-squared Value', range: [0.5, 0.8] },
        height: 350
    };
    Plotly.newPlot('chart-r2-compare', [trace], layout, {responsive: true, displayModeBar: false});
}

// 4. Predicted vs Actual (Scatter Plot)
function drawPredVsActual(data) {
    const trace = {
        x: data.actual,
        y: data.predicted,
        mode: 'markers',
        type: 'scatter',
        name: 'Data Points',
        text: data.years,
        hovertemplate: 'Year: %{text}<br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>',
        marker: { color: '#007bff', opacity: 0.7, size: 8 }
    };

    const line = {
        x: [0.2, 0.8], 
        y: [0.2, 0.8],
        mode: 'lines',
        type: 'scatter',
        name: 'Ideal Fit (Y=X)',
        line: { color: 'black', dash: 'dash' }
    };

    const layout = {
        title: { text: 'Predicted vs. Actual (Botswana Data)' },
        xaxis: { title: 'Actual Formal Saving Rate' },
        yaxis: { title: 'Predicted Formal Saving Rate' },
        height: 350
    };
    Plotly.newPlot('chart-pred-actual', [trace, line], layout, {responsive: true, displayModeBar: true});
}


// 6. Correlation (Bar Chart)
function drawCorrelation(data) {
    const features = data.map(d => d.feature).reverse();
    const correlation = data.map(d => d.correlation).reverse();

    const trace = {
        x: correlation,
        y: features,
        type: 'bar',
        orientation: 'h',
        marker: { color: correlation.map(c => c > 0 ? '#17a2b8' : '#6c757d') }
    };

    const layout = {
        title: { text: 'Correlation with Formal Saving' },
        margin: { l: 150, r: 20, t: 50, b: 50 },
        xaxis: { title: 'Correlation Coefficient' },
        height: 350
    };
    Plotly.newPlot('chart-correlation', [trace], layout, {responsive: true, displayModeBar: false});
}

// 8. Country Comparison (Scatter Plot)
function drawCountryComparison(data) {
    const countries = data.map(d => d.country);
    const mobile_pen = data.map(d => d.mobile_pen * 100);
    const formal_saving = data.map(d => d.formal_saving * 100);

    const trace = {
        x: mobile_pen,
        y: formal_saving,
        mode: 'markers',
        type: 'scatter',
        text: countries,
        hovertemplate: '<b>%{text}</b><br>Mobile Pen: %{x:.1f}%<br>Formal Saving: %{y:.1f}%<extra></extra>',
        marker: { size: 12, color: '#dc3545' }
    };

    const layout = {
        title: { text: 'Top Predictor vs. Target (Sample Countries)' },
        xaxis: { title: 'Mobile Penetration (%)' },
        yaxis: { title: 'Formal Saving Rate (%)' },
        height: 350
    };
    Plotly.newPlot('chart-country-comp', [trace], layout, {responsive: true, displayModeBar: true});
}

// 7. Gauge Chart
function drawGauge(stability_score) {
    const max_target = 80; // Goal R2 percentage
    
    const gauge = {
        type: "indicator",
        mode: "gauge+number",
        value: stability_score,
        title: { text: "R² Score (%)" },
        gauge: {
            axis: { range: [null, max_target], tickwidth: 1, tickcolor: "#1a5e82" },
            bar: { color: "#1a5e82" },
            bgcolor: "white",
            steps: [
                { range: [0, 50], color: "red" },
                { range: [50, 70], color: "yellow" },
                { range: [70, max_target], color: "lightgreen" }
            ],
            threshold: {
                line: { color: "red", width: 4 },
                thickness: 0.75,
                value: 71.41 // Ensemble R2 in percentage
            }
        }
    };
    
    const layout = { margin: { t: 0, b: 0, l: 30, r: 30 }, height: 350 };
    Plotly.newPlot('chart-gauge', [gauge], layout, {responsive: true, displayModeBar: false});
}
