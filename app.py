from flask import Flask, render_template, request, send_file, jsonify
import numpy as np
import joblib
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from datetime import datetime
import os
import json
import pandas as pd

print("App is starting...")

# Initialize Flask app
app = Flask(__name__)

# Try to load model pipeline if available
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'fraud_pipeline.pkl')
PREDICTIONS_PATH = os.path.join(os.path.dirname(__file__), 'predictions.json')
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'insurance_claims.csv')
pipeline = None
model = None
scaler = None
feature_columns = []
risk_thresholds = {'low_upper': 40.0, 'high_lower': 70.0}
feature_ranges = {}
if os.path.exists(MODEL_PATH):
    try:
        pipeline = joblib.load(MODEL_PATH)
        model = pipeline.get('model')
        scaler = pipeline.get('scaler')
        feature_columns = pipeline.get('feature_columns', [])
        loaded_thresholds = pipeline.get('risk_thresholds') or {}
        if isinstance(loaded_thresholds, dict):
            risk_thresholds['low_upper'] = float(loaded_thresholds.get('low_upper', 40.0))
            risk_thresholds['high_lower'] = float(loaded_thresholds.get('high_lower', 70.0))
            # Keep thresholds in a valid order to avoid broken labels.
            if risk_thresholds['high_lower'] <= risk_thresholds['low_upper']:
                risk_thresholds = {'low_upper': 40.0, 'high_lower': 70.0}
        loaded_ranges = pipeline.get('feature_ranges') or {}
        if isinstance(loaded_ranges, dict):
            for col, bounds in loaded_ranges.items():
                if isinstance(bounds, dict) and 'min' in bounds and 'max' in bounds:
                    feature_ranges[col] = {
                        'min': float(bounds['min']),
                        'max': float(bounds['max']),
                    }
        print('Loaded fraud_pipeline.pkl')
    except Exception as e:
        print('Failed loading pipeline:', e)
else:
    print('fraud_pipeline.pkl not found; prediction will be disabled')


def _load_feature_ranges_from_csv():
    if not feature_columns or not os.path.exists(DATASET_PATH):
        return
    try:
        df = pd.read_csv(DATASET_PATH)
        for col in feature_columns:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            if series.empty:
                continue
            feature_ranges[col] = {'min': float(series.min()), 'max': float(series.max())}
    except Exception as e:
        print('Failed loading feature ranges from CSV:', e)


# Fallback: derive ranges from training data if not saved in pipeline.
if not feature_ranges:
    _load_feature_ranges_from_csv()


def _get_positive_class_index(estimator):
    classes = list(getattr(estimator, 'classes_', []))
    if not classes:
        return 1
    for candidate in (1, 'Y', 'y', True):
        if candidate in classes:
            return classes.index(candidate)
    return 1 if len(classes) > 1 else 0


def _risk_label(probability):
    low_upper = float(risk_thresholds.get('low_upper', 40.0))
    high_lower = float(risk_thresholds.get('high_lower', 70.0))
    if probability < low_upper:
        return 'Low Risk'
    if probability >= high_lower:
        return 'High Risk'
    return 'Medium Risk'


def _sanitize_input(col, raw_value):
    numeric_val = float(raw_value)
    bounds = feature_ranges.get(col)
    if not bounds:
        return numeric_val, None
    low = float(bounds['min'])
    high = float(bounds['max'])
    clipped = min(max(numeric_val, low), high)
    if clipped != numeric_val:
        return clipped, f"{col} adjusted from {numeric_val:g} to trained range [{low:g}, {high:g}]"
    return clipped, None


# ==============================
# Routes
# ==============================
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET'])
def predict_page():
    # Render the prediction form page
    return render_template('predict.html', risk_thresholds=risk_thresholds)


@app.route('/predict.html', methods=['GET'])
def predict_page_html():
    # Alias to support the landing page's hard link to predict.html
    return render_template('predict.html', risk_thresholds=risk_thresholds)


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    # Handle form POST, compute prediction, and render predict.html with results
    if model is None or scaler is None or not feature_columns:
        # Model not available - show an error message in the same page
        return render_template('predict.html', probability=None, error='Model not available', risk_thresholds=risk_thresholds)

    try:
        data = {}
        input_warnings = []
        for col in feature_columns:
            # fall back to 0 if field missing
            val = request.form.get(col, request.form.get(col, 0))
            cleaned, warn = _sanitize_input(col, val)
            data[col] = cleaned
            if warn:
                input_warnings.append(warn)

        final_features = np.array([[data[col] for col in feature_columns]])
        final_features = scaler.transform(final_features)

        pos_idx = _get_positive_class_index(model)
        probability = model.predict_proba(final_features)[0][pos_idx] * 100
        probability = round(float(probability), 2)
        risk = _risk_label(probability)

        # Build a record for persistence and dashboard display
        try:
            record = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'probability': probability,
                'risk': risk,
            }
            # include submitted numeric fields where present
            for f in ['months_as_customer','age','policy_annual_premium','umbrella_limit','total_claim_amount','injury_claim','property_claim','vehicle_claim']:
                record[f] = float(request.form.get(f, 0)) if request.form.get(f, '') != '' else None

            # load existing predictions
            preds = []
            if os.path.exists(PREDICTIONS_PATH):
                try:
                    with open(PREDICTIONS_PATH, 'r', encoding='utf-8') as fh:
                        preds = json.load(fh) or []
                except Exception:
                    preds = []

            preds.insert(0, record)
            # keep most recent 200
            preds = preds[:200]
            with open(PREDICTIONS_PATH, 'w', encoding='utf-8') as fh:
                json.dump(preds, fh, indent=2, ensure_ascii=False)
        except Exception as e:
            print('Failed saving prediction:', e)

        return render_template(
            'predict.html',
            probability=probability,
            risk=risk,
            form_data=request.form,
            input_warnings=input_warnings,
            risk_thresholds=risk_thresholds
        )

    except Exception as e:
        return render_template('predict.html', probability=None, error=str(e), risk_thresholds=risk_thresholds)


@app.route('/investigation', methods=['GET', 'POST'])
def raise_investigation():
    if request.method == 'POST':
        # Accept JSON or form data
        data = {}
        try:
            data = request.get_json() or {}
        except Exception:
            data = {}
        if not data:
            data = request.form.to_dict()

        raised_by = data.get('raised_by', 'Unknown')
        description = data.get('description', '')

        # In a real app we'd persist this to a DB or ticketing system.
        print(f"Investigation raised by: {raised_by}\nDescription: {description}")

        return jsonify({ 'status': 'success', 'message': f'Investigation raised by {raised_by}' })

    # GET fallback: simple message
    return "<h2>🚨 Investigation Case Created Successfully!</h2>"


def _load_predictions():
    try:
        if os.path.exists(PREDICTIONS_PATH):
            with open(PREDICTIONS_PATH, 'r', encoding='utf-8') as fh:
                return json.load(fh) or []
    except Exception:
        return []
    return []


@app.route('/dashboard', methods=['GET'])
def dashboard():
    preds = _load_predictions()
    stats = {
        'total_analyses': 0,
        'high_risk_count': 0,
        'medium_risk_count': 0,
        'low_risk_count': 0,
        'avg_probability': 0
    }
    latest = None
    if preds:
        stats['total_analyses'] = len(preds)
        stats['high_risk_count'] = sum(1 for p in preds if p.get('risk') == 'High Risk')
        stats['medium_risk_count'] = sum(1 for p in preds if p.get('risk') == 'Medium Risk')
        stats['low_risk_count'] = sum(1 for p in preds if p.get('risk') == 'Low Risk')
        avg = sum((p.get('probability', 0) or 0) for p in preds) / max(1, len(preds))
        stats['avg_probability'] = round(avg, 1)
        latest = preds[0]

    return render_template('dashboard.html', latest=latest, stats=stats, predictions=preds)


@app.route('/download_pdf')
def download_report():
    # Read available parameters (front-end supplies them as query params)
    args = request.args
    probability = args.get('probability', 'N/A')
    risk = args.get('risk', 'N/A')
    insights_raw = args.get('insights', '')

    # Known form fields we expect; include any that were submitted
    known_fields = {
        'total_claim_amount': 'Total Claim Amount',
        'injury_claim': 'Injury Claim',
        'property_claim': 'Property Claim',
        'vehicle_claim': 'Vehicle Claim',
        'policy_annual_premium': 'Annual Premium',
        'age': 'Insured Age'
    }

    # Helper formatting
    def parse_prob(pstr):
        try:
            return float(str(pstr).replace('%',''))
        except Exception:
            return None

    def money(v):
        try:
            f = float(v)
            return f"${f:,.2f}"
        except Exception:
            return str(v)

    def wrap_text_to_width(text, font_name, font_size, max_width, max_lines=2):
        words = str(text).split()
        if not words:
            return ['']
        lines = []
        current = words[0]
        for word in words[1:]:
            trial = current + ' ' + word
            if pdfmetrics.stringWidth(trial, font_name, font_size) <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            while pdfmetrics.stringWidth(lines[-1] + '...', font_name, font_size) > max_width and lines[-1]:
                lines[-1] = lines[-1][:-1]
            lines[-1] = lines[-1] + '...'
        return lines

    prob_num = parse_prob(probability)
    # Determine verdict and fraud type heuristics
    verdict = 'N/A'
    risk_normalized = str(risk).strip().lower()
    if risk_normalized.startswith('high'):
        verdict = 'FRAUDULENT CLAIM'
    elif risk_normalized.startswith('low'):
        verdict = 'LEGITIMATE CLAIM'
    elif risk_normalized.startswith('medium'):
        verdict = 'REVIEW RECOMMENDED'
    elif prob_num is None:
        verdict = 'N/A'
    else:
        verdict = 'FRAUDULENT CLAIM' if prob_num >= float(risk_thresholds.get('high_lower', 70.0)) else 'LEGITIMATE CLAIM'

    fraud_type = 'Pattern Anomaly' if ('pattern' in insights_raw.lower()) else 'N/A'

    # Create PDF
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header
    p.setFillColorRGB(0.06, 0.75, 0.98)  # cyan-like
    p.setFont('Helvetica-Bold', 22)
    title = 'SmartSecure AI - Fraud Detection Report'
    p.drawCentredString(width/2, height - 50, title)
    p.setFillColorRGB(0,0,0)

    # Timestamp
    p.setFont('Helvetica', 9)
    p.drawString(72, height - 80, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y = height - 110

    # Prediction Summary
    p.setFillColorRGB(0.06, 0.75, 0.98)
    p.setFont('Helvetica-Bold', 14)
    p.drawString(72, y, 'Prediction Summary')
    p.setFillColorRGB(0,0,0)
    y -= 20
    p.setFont('Helvetica', 11)
    p.drawString(72, y, verdict)
    y -= 28

    # Risk Assessment Table
    p.setFillColorRGB(0.06, 0.75, 0.98)
    p.setFont('Helvetica-Bold', 14)
    p.drawString(72, y, 'Risk Assessment')
    y -= 18

    table_x = 180
    table_w = 260
    col1_w = 140
    col2_w = table_w - col1_w
    row_h = 22

    # Header row
    p.setFillColor(colors.cyan)
    p.rect(table_x, y - row_h, table_w, row_h, fill=1, stroke=1)
    p.setFillColorRGB(0,0,0)
    p.setFont('Helvetica-Bold', 10)
    p.drawString(table_x + 8, y - 16, 'Metric')
    p.drawString(table_x + col1_w + 8, y - 16, 'Value')
    y -= row_h

    # Rows
    rows = [
        ('Fraud Probability', f"{probability}"),
        ('Risk Level', risk),
        ('Fraud Type', fraud_type)
    ]
    p.setFont('Helvetica', 10)
    for label, val in rows:
        wrapped_value = wrap_text_to_width(val, 'Helvetica', 10, col2_w - 16, max_lines=2)
        this_row_h = max(row_h, 8 + (len(wrapped_value) * 12))
        p.setFillColorRGB(0.96,0.96,0.86)
        p.rect(table_x, y - this_row_h, table_w, this_row_h, fill=1, stroke=1)
        p.setFillColorRGB(0,0,0)
        p.drawString(table_x + 8, y - 16, label)
        for line_idx, line in enumerate(wrapped_value):
            p.drawString(table_x + col1_w + 8, y - 16 - (line_idx * 12), line)
        y -= this_row_h

    y -= 18

    # Explainable AI Analysis
    p.setFillColorRGB(0.06, 0.75, 0.98)
    p.setFont('Helvetica-Bold', 14)
    p.drawString(72, y, 'Explainable AI Analysis')
    y -= 18
    p.setFillColorRGB(0,0,0)
    p.setFont('Helvetica', 10)
    if insights_raw:
        insights = insights_raw.split(' || ')
        for ins in insights:
            # bullet
            p.drawString(80, y, u'• ' + ins)
            y -= 14
            if y < 120:
                p.showPage()
                y = height - 72
    else:
        p.drawString(80, y, 'No insights available')
        y -= 14

    y -= 12

    # Claim Details table
    p.setFillColorRGB(0.06, 0.75, 0.98)
    p.setFont('Helvetica-Bold', 14)
    p.drawString(72, y, 'Claim Details')
    y -= 18

    # Table layout
    table_x = 180
    table_w = 260
    col1_w = 160
    col2_w = table_w - col1_w
    row_h = 20

    # Header
    p.setFillColor(colors.cyan)
    p.rect(table_x, y - row_h, table_w, row_h, fill=1, stroke=1)
    p.setFillColorRGB(0,0,0)
    p.setFont('Helvetica-Bold', 10)
    p.drawString(table_x + 8, y - 15, 'Field')
    p.drawString(table_x + col1_w + 8, y - 15, 'Value')
    y -= row_h

    p.setFont('Helvetica', 10)
    for key, label in known_fields.items():
        val = args.get(key, '')
        display = money(val) if key != 'age' else (str(int(float(val))) if val not in (None, '') else '')
        p.setFillColorRGB(0.96,0.96,0.86)
        p.rect(table_x, y - row_h, table_w, row_h, fill=1, stroke=1)
        p.setFillColorRGB(0,0,0)
        p.drawString(table_x + 8, y - 15, label)
        p.drawString(table_x + col1_w + 8, y - 15, display)
        y -= row_h
        if y < 120:
            p.showPage()
            y = height - 72

    p.showPage()
    p.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="fraud_report.pdf",
        mimetype="application/pdf"
    )


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)