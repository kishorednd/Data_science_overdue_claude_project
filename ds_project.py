"""End-to-end data science project for overdue prediction dataset.
Implemented with Python standard library only (no external dependencies).
"""

import csv
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

random.seed(42)

DATA_PATH = Path("overdue_prediction_dataset.csv")
OUT_DIR = Path("artifacts")
OUT_DIR.mkdir(exist_ok=True)

NUMERIC_COLS = [
    "Age",
    "Annual_Income",
    "Credit_Score",
    "Loan_Amount",
    "Loan_Term_Months",
    "Interest_Rate",
    "Monthly_Payment",
    "Account_Age_Months",
    "Number_of_Existing_Loans",
    "Number_of_Late_Payments",
    "Months_Since_Last_Payment",
    "Debt_to_Income_Ratio",
]
CATEGORICAL_COLS = ["Gender", "City", "Employment_Status"]
TARGET_COL = "Overdue_Status"


def parse_float(v):
    if v is None:
        return None
    v = str(v).strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def normalize_category(col, val):
    v = (val or "").strip().lower()
    if v in {"", "unknown", "n/a", "na"}:
        return "missing"
    if col == "Gender":
        if v in {"m", "male"}:
            return "male"
        if v in {"f", "female"}:
            return "female"
        return "other"
    if col == "Employment_Status":
        if v in {"employed"}:
            return "employed"
        if v in {"unemployed"}:
            return "unemployed"
        if v in {"self-employed", "self employed", "self_employed"}:
            return "self-employed"
        return v
    if col == "City":
        mapping = {
            "new york": "new york",
            "los angeles": "los angeles",
            "chicago": "chicago",
            "houston": "houston",
            "phoenix": "phoenix",
        }
        return mapping.get(v, v)
    return v


def parse_date_features(date_str):
    ds = (date_str or "").strip()
    if not ds:
        return {"app_year": None, "app_month": None, "app_weekday": None}
    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]:
        try:
            dt = datetime.strptime(ds, fmt)
            return {
                "app_year": float(dt.year),
                "app_month": float(dt.month),
                "app_weekday": float(dt.weekday()),
            }
        except ValueError:
            continue
    return {"app_year": None, "app_month": None, "app_weekday": None}


def read_data():
    rows = []
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {}
            for c in NUMERIC_COLS:
                row[c] = parse_float(raw.get(c))
            for c in CATEGORICAL_COLS:
                row[c] = normalize_category(c, raw.get(c))
            row.update(parse_date_features(raw.get("Application_Date")))
            row[TARGET_COL] = parse_float(raw.get(TARGET_COL))
            row["Customer_ID"] = raw.get("Customer_ID", "")
            rows.append(row)
    return rows


def infer_problem_type(rows):
    vals = [r[TARGET_COL] for r in rows if r[TARGET_COL] is not None]
    unique = sorted(set(vals))
    if len(unique) <= 10 and all(v in {0.0, 1.0} for v in unique):
        return "binary_classification", unique
    if len(unique) <= 20:
        return "multiclass_classification", unique
    return "regression", unique


def missing_report(rows, cols):
    rep = {}
    n = len(rows)
    for c in cols:
        miss = sum(1 for r in rows if r[c] is None or r[c] == "missing")
        rep[c] = (miss, miss / n)
    return rep


def impute(rows):
    numeric_all = NUMERIC_COLS + ["app_year", "app_month", "app_weekday"]
    medians = {}
    for c in numeric_all:
        vals = [r[c] for r in rows if r[c] is not None]
        medians[c] = statistics.median(vals) if vals else 0.0
    cat_modes = {}
    for c in CATEGORICAL_COLS:
        vals = [r[c] for r in rows if r[c] != "missing"]
        cat_modes[c] = Counter(vals).most_common(1)[0][0] if vals else "missing"

    for r in rows:
        for c in numeric_all:
            if r[c] is None:
                r[c] = medians[c]
        for c in CATEGORICAL_COLS:
            if r[c] == "missing":
                r[c] = cat_modes[c]
        if r[TARGET_COL] is None:
            r[TARGET_COL] = 0.0
    return medians, cat_modes


def winsorize_iqr(rows, cols, whisker=1.5):
    caps = {}
    for c in cols:
        vals = sorted(r[c] for r in rows)
        q1 = vals[int(0.25 * (len(vals) - 1))]
        q3 = vals[int(0.75 * (len(vals) - 1))]
        iqr = q3 - q1
        lo = q1 - whisker * iqr
        hi = q3 + whisker * iqr
        caps[c] = (lo, hi)
        for r in rows:
            if r[c] < lo:
                r[c] = lo
            elif r[c] > hi:
                r[c] = hi
    return caps


def feature_engineer(rows):
    for r in rows:
        income = max(r["Annual_Income"], 1e-6)
        r["Loan_to_Income"] = r["Loan_Amount"] / income
        r["Payment_to_Income"] = (r["Monthly_Payment"] * 12.0) / income
        r["Late_per_Loan"] = r["Number_of_Late_Payments"] / (1.0 + r["Number_of_Existing_Loans"])
        r["Credit_Income_Interaction"] = r["Credit_Score"] * math.log1p(income)


def train_test_split(rows, test_size=0.2):
    idx = list(range(len(rows)))
    random.shuffle(idx)
    cut = int(len(rows) * (1 - test_size))
    train_idx = set(idx[:cut])
    train = [rows[i] for i in range(len(rows)) if i in train_idx]
    test = [rows[i] for i in range(len(rows)) if i not in train_idx]
    return train, test


def build_feature_space(train_rows, test_rows):
    numeric = NUMERIC_COLS + ["app_year", "app_month", "app_weekday", "Loan_to_Income", "Payment_to_Income", "Late_per_Loan", "Credit_Income_Interaction"]
    cat_values = {c: sorted(set(r[c] for r in train_rows)) for c in CATEGORICAL_COLS}

    means, stds = {}, {}
    for c in numeric:
        vals = [r[c] for r in train_rows]
        means[c] = sum(vals) / len(vals)
        var = sum((x - means[c]) ** 2 for x in vals) / len(vals)
        stds[c] = math.sqrt(var) if var > 1e-12 else 1.0

    def encode(rows):
        X, y, ids = [], [], []
        for r in rows:
            vec = []
            for c in numeric:
                vec.append((r[c] - means[c]) / stds[c])
            for c in CATEGORICAL_COLS:
                cats = cat_values[c]
                base = cats[0] if cats else None
                for cv in cats[1:]:
                    vec.append(1.0 if r[c] == cv else 0.0)
            X.append(vec)
            y.append(int(r[TARGET_COL]))
            ids.append(r["Customer_ID"])
        return X, y, ids

    feature_names = list(numeric)
    for c in CATEGORICAL_COLS:
        cats = cat_values[c]
        for cv in cats[1:]:
            feature_names.append(f"{c}__{cv}")

    return encode(train_rows), encode(test_rows), feature_names


def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1 / (1 + ez)
    ez = math.exp(z)
    return ez / (1 + ez)


def train_logistic(X, y, lr=0.05, epochs=180, l2=0.001):
    n, d = len(X), len(X[0])
    w = [0.0] * d
    b = 0.0
    for _ in range(epochs):
        grad_w = [0.0] * d
        grad_b = 0.0
        for xi, yi in zip(X, y):
            p = sigmoid(sum(wj * xj for wj, xj in zip(w, xi)) + b)
            err = p - yi
            for j in range(d):
                grad_w[j] += err * xi[j]
            grad_b += err
        for j in range(d):
            grad_w[j] = grad_w[j] / n + l2 * w[j]
            w[j] -= lr * grad_w[j]
        b -= lr * grad_b / n
    return w, b


def predict_logistic(X, model):
    w, b = model
    probs = [sigmoid(sum(wj * xj for wj, xj in zip(w, xi)) + b) for xi in X]
    preds = [1 if p >= 0.5 else 0 for p in probs]
    return preds, probs


def train_gaussian_nb(X, y):
    classes = sorted(set(y))
    prior = {}
    mean = {c: [0.0] * len(X[0]) for c in classes}
    var = {c: [1.0] * len(X[0]) for c in classes}
    for c in classes:
        Xc = [x for x, t in zip(X, y) if t == c]
        prior[c] = len(Xc) / len(X)
        for j in range(len(X[0])):
            m = sum(x[j] for x in Xc) / len(Xc)
            v = sum((x[j] - m) ** 2 for x in Xc) / len(Xc)
            mean[c][j] = m
            var[c][j] = max(v, 1e-6)
    return prior, mean, var


def predict_gaussian_nb(X, model):
    prior, mean, var = model
    classes = sorted(prior)
    preds, probs = [], []
    for x in X:
        logp = {}
        for c in classes:
            s = math.log(prior[c])
            for j, xv in enumerate(x):
                m, v = mean[c][j], var[c][j]
                s += -0.5 * math.log(2 * math.pi * v) - ((xv - m) ** 2) / (2 * v)
            logp[c] = s
        maxlog = max(logp.values())
        exps = {c: math.exp(v - maxlog) for c, v in logp.items()}
        total = sum(exps.values())
        p1 = exps.get(1, 0.0) / total
        probs.append(p1)
        preds.append(1 if p1 >= 0.5 else 0)
    return preds, probs


def train_knn(X, y, k=7, max_points=600):
    # Subsample for computational tractability in pure-python implementation.
    if len(X) > max_points:
        idx = list(range(len(X)))
        random.shuffle(idx)
        idx = idx[:max_points]
        Xs = [X[i] for i in idx]
        ys = [y[i] for i in idx]
        return Xs, ys, k
    return X, y, k


def predict_knn(X, model):
    Xtr, ytr, k = model
    preds, probs = [], []
    for x in X:
        dists = []
        for xt, yt in zip(Xtr, ytr):
            d = sum((a - b) ** 2 for a, b in zip(x, xt))
            dists.append((d, yt))
        dists.sort(key=lambda z: z[0])
        nbrs = dists[:k]
        p1 = sum(1 for _, t in nbrs if t == 1) / k
        probs.append(p1)
        preds.append(1 if p1 >= 0.5 else 0)
    return preds, probs


def metrics(y_true, y_pred, y_prob):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    pairs = sorted(zip(y_prob, y_true), key=lambda x: x[0], reverse=True)
    p = sum(y_true)
    n = len(y_true) - p
    tpr_prev = fpr_prev = auc = 0.0
    tp_c = fp_c = 0
    for _, yt in pairs:
        if yt == 1:
            tp_c += 1
        else:
            fp_c += 1
        tpr = tp_c / p if p else 0
        fpr = fp_c / n if n else 0
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        tpr_prev, fpr_prev = tpr, fpr

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def baseline_predict(y_train, n):
    maj = 1 if sum(y_train) >= (len(y_train) / 2) else 0
    return [maj] * n, [float(maj)] * n


def tune_logistic(Xtr, ytr, Xval, yval):
    grid = [(0.03, 120), (0.05, 180), (0.08, 220)]
    best = None
    for lr, ep in grid:
        m = train_logistic(Xtr, ytr, lr=lr, epochs=ep)
        pred, proba = predict_logistic(Xval, m)
        s = metrics(yval, pred, proba)["f1"]
        if best is None or s > best[0]:
            best = (s, lr, ep, m)
    return best


def ascii_hist(values, bins=20, width=40):
    mn, mx = min(values), max(values)
    if mx - mn < 1e-9:
        return ["all values equal"]
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int((v - mn) / (mx - mn) * bins))
        counts[idx] += 1
    mxc = max(counts)
    lines = []
    for i, c in enumerate(counts):
        left = mn + (mx - mn) * i / bins
        right = mn + (mx - mn) * (i + 1) / bins
        bar = "#" * int(width * c / mxc)
        lines.append(f"{left:8.2f} - {right:8.2f} | {bar} ({c})")
    return lines


def save_svg_bar(path, title, labels, values):
    w, h = 800, 420
    margin = 60
    plot_w = w - 2 * margin
    plot_h = h - 2 * margin
    max_v = max(values) if values else 1
    bar_w = plot_w / max(1, len(values))

    rects = []
    texts = []
    for i, (lab, v) in enumerate(zip(labels, values)):
        bh = 0 if max_v == 0 else plot_h * (v / max_v)
        x = margin + i * bar_w + 5
        y = margin + (plot_h - bh)
        rects.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-10:.1f}" height="{bh:.1f}" fill="#4C78A8"/>')
        texts.append(f'<text x="{x+2:.1f}" y="{h-margin+15}" font-size="10" transform="rotate(25 {x+2:.1f},{h-margin+15})">{lab}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
    <rect width="100%" height="100%" fill="white"/>
    <text x="{w/2}" y="30" text-anchor="middle" font-size="20">{title}</text>
    <line x1="{margin}" y1="{h-margin}" x2="{w-margin}" y2="{h-margin}" stroke="black"/>
    <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{h-margin}" stroke="black"/>
    {''.join(rects)}
    {''.join(texts)}
</svg>'''
    Path(path).write_text(svg, encoding="utf-8")


def main():
    rows = read_data()
    problem_type, target_unique = infer_problem_type(rows)

    miss_before = missing_report(rows, NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL, "app_year"])
    medians, modes = impute(rows)
    caps = winsorize_iqr(rows, NUMERIC_COLS + ["app_year", "app_month", "app_weekday"])
    feature_engineer(rows)

    overdue_rate = sum(int(r[TARGET_COL]) for r in rows) / len(rows)

    train_rows, test_rows = train_test_split(rows)
    tr_pack, te_pack, feature_names = build_feature_space(train_rows, test_rows)
    X_train, y_train, _ = tr_pack
    X_test, y_test, test_ids = te_pack

    split = int(0.8 * len(X_train))
    Xtr, ytr = X_train[:split], y_train[:split]
    Xval, yval = X_train[split:], y_train[split:]

    tuned = tune_logistic(Xtr, ytr, Xval, yval)
    _, best_lr, best_ep, _ = tuned

    models = {}
    models["BaselineMajority"] = (None, baseline_predict)
    models["LogisticRegression"] = (train_logistic(X_train, y_train, lr=best_lr, epochs=best_ep), predict_logistic)
    models["GaussianNB"] = (train_gaussian_nb(X_train, y_train), predict_gaussian_nb)
    models["KNN"] = (train_knn(X_train, y_train, k=7), predict_knn)

    results = []
    predictions_store = {}
    for name, (model, pred_fn) in models.items():
        if name == "BaselineMajority":
            pred, prob = baseline_predict(y_train, len(y_test))
        else:
            pred, prob = pred_fn(X_test, model)
        m = metrics(y_test, pred, prob)
        m["model"] = name
        results.append(m)
        predictions_store[name] = (pred, prob)

    results.sort(key=lambda x: x["f1"], reverse=True)
    best_model_name = results[0]["model"]

    if best_model_name == "BaselineMajority":
        best_pred, best_prob = baseline_predict(y_train, len(y_test))
    else:
        best_pred, best_prob = models[best_model_name][1](X_test, models[best_model_name][0])

    pred_path = OUT_DIR / "test_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Customer_ID", "Actual", "Predicted", "Probability_Overdue", "Risk_Band"])
        for cid, yt, yp, p in zip(test_ids, y_test, best_pred, best_prob):
            band = "High" if p >= 0.70 else ("Medium" if p >= 0.40 else "Low")
            w.writerow([cid, yt, yp, f"{p:.4f}", band])

    feat_importance = []
    if best_model_name == "LogisticRegression":
        w, _ = models["LogisticRegression"][0]
        feat_importance = sorted(zip(feature_names, [abs(x) for x in w]), key=lambda x: x[1], reverse=True)[:15]
    else:
        corrs = []
        for j, fn in enumerate(feature_names):
            x = [row[j] for row in X_train]
            mx = sum(x) / len(x)
            my = sum(y_train) / len(y_train)
            cov = sum((a - mx) * (b - my) for a, b in zip(x, y_train)) / len(x)
            vx = sum((a - mx) ** 2 for a in x) / len(x)
            vy = sum((b - my) ** 2 for b in y_train) / len(x)
            corr = cov / math.sqrt(vx * vy) if vx > 1e-12 and vy > 1e-12 else 0.0
            corrs.append((fn, abs(corr)))
        feat_importance = sorted(corrs, key=lambda z: z[1], reverse=True)[:15]

    with (OUT_DIR / "model_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy", "precision", "recall", "f1", "roc_auc", "tp", "tn", "fp", "fn"])
        for r in results:
            w.writerow([r["model"], f"{r['accuracy']:.4f}", f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}", f"{r['roc_auc']:.4f}", r["tp"], r["tn"], r["fp"], r["fn"]])

    city_counts = Counter(r["City"] for r in rows)
    save_svg_bar(OUT_DIR / "city_distribution.svg", "Customer Distribution by City", list(city_counts.keys()), list(city_counts.values()))
    emp_overdue = defaultdict(lambda: [0, 0])
    for r in rows:
        e = r["Employment_Status"]
        emp_overdue[e][0] += int(r[TARGET_COL])
        emp_overdue[e][1] += 1
    labels = list(emp_overdue.keys())
    rates = [v[0] / v[1] for v in emp_overdue.values()]
    save_svg_bar(OUT_DIR / "overdue_by_employment.svg", "Overdue Rate by Employment Status", labels, rates)

    report_lines = []
    report_lines.append("# End-to-End Data Science Project: Overdue Payment Prediction\n")
    report_lines.append("## 1. Problem Understanding")
    report_lines.append(f"- **Business objective:** predict if a customer will become overdue so collections and risk teams can prioritize intervention.")
    report_lines.append(f"- **Detected problem type:** `{problem_type}`.")
    report_lines.append(f"- **Target variable selected:** `{TARGET_COL}` with classes {target_unique}.")
    report_lines.append(f"- **Target prevalence:** overdue rate is **{overdue_rate:.2%}** across {len(rows):,} records.")

    report_lines.append("\n## 2. Data Exploration")
    report_lines.append(f"- Dataset size: **{len(rows)} rows x {len(NUMERIC_COLS)+len(CATEGORICAL_COLS)+5} prepared columns**.")
    report_lines.append("- Missing values identified before imputation:")
    for c, (m, r) in miss_before.items():
        report_lines.append(f"  - {c}: {m} ({r:.2%})")
    report_lines.append("- Visual artifacts generated:")
    report_lines.append("  - `artifacts/city_distribution.svg`")
    report_lines.append("  - `artifacts/overdue_by_employment.svg`")
    report_lines.append("- Distribution snapshot (Debt_to_Income_Ratio histogram):")
    for ln in ascii_hist([r["Debt_to_Income_Ratio"] for r in rows], bins=12, width=30)[:12]:
        report_lines.append(f"  - `{ln}`")

    report_lines.append("\n## 3. Data Cleaning & Preparation")
    report_lines.append("- Standardized categorical inconsistencies (e.g., `MALE`, `M`, `male` unified).")
    report_lines.append("- Parsed mixed date formats into numeric year/month/weekday features.")
    report_lines.append("- Imputed numeric nulls with medians and categorical nulls with modes.")
    report_lines.append("- Capped numeric outliers using IQR winsorization.")
    report_lines.append("- One-hot encoded categorical features and standardized numeric fields.")

    report_lines.append("\n## 4. Feature Engineering")
    report_lines.append("- Added `Loan_to_Income`, `Payment_to_Income`, `Late_per_Loan`, `Credit_Income_Interaction`.")
    report_lines.append("- These features represent affordability stress and repayment behavior signals.")

    report_lines.append("\n## 5. Model Training")
    report_lines.append("- Split data into train/test (80/20).")
    report_lines.append("- Trained models: Baseline majority, Logistic Regression, Gaussian Naive Bayes, KNN.")
    report_lines.append(f"- Logistic Regression hyperparameter search over learning rate/epochs selected `lr={best_lr}`, `epochs={best_ep}` based on validation F1.")

    report_lines.append("\n## 6. Model Evaluation")
    report_lines.append("- Evaluated using Accuracy, Precision, Recall, F1, ROC-AUC to balance false positives and false negatives.")
    report_lines.append(f"- Best model: **{best_model_name}** (sorted by F1).")

    report_lines.append("\n## 7. Model Comparison")
    report_lines.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |")
    report_lines.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        report_lines.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} |")

    report_lines.append("\n## 8. Predictions")
    report_lines.append("- Generated test-set predictions and risk bands in `artifacts/test_predictions.csv`.")
    report_lines.append("- Risk bands: High (>=0.70), Medium (0.40-0.69), Low (<0.40).")

    report_lines.append("\n## 9. Insights & Business Conclusion")
    report_lines.append("- Main risk drivers include late payment behavior, affordability-related ratios, and loan burden variables.")
    report_lines.append("- Use predicted risk tiers to prioritize proactive reminders, restructuring offers, and collections workflows.")
    report_lines.append("- Track ROI by comparing recovered amounts and prevented delinquencies across intervention strategies.")

    report_lines.append("\n### Feature Importance (Top 10)")
    for n, v in feat_importance[:10]:
        report_lines.append(f"- {n}: {v:.4f}")

    report_lines.append("\n### Risks, Limitations, and Improvements")
    report_lines.append("- Synthetic/limited feature space may not capture all delinquency drivers (behavioral, macroeconomic, channel interactions).")
    report_lines.append("- Monitor for data drift and threshold drift each month; recalibrate cutoffs by business capacity.")
    report_lines.append("- Next iteration: gradient boosting, probability calibration, reject inference, and fairness diagnostics.")

    report_lines.append("\n## 10. Deployment & Monitoring Strategy")
    report_lines.append("- Batch score new applications daily and push risk band to CRM/collections queue.")
    report_lines.append("- Operational metrics: model AUC/F1, capture rate in top risk decile, intervention conversion, and false-positive cost.")
    report_lines.append("- Governance: champion-challenger retraining every quarter or when drift thresholds are breached.")

    report_lines.append("\n## Appendix: Preprocessing Parameters")
    report_lines.append("- Numeric medians used for imputation:")
    for c, v in medians.items():
        report_lines.append(f"  - {c}: {v:.4f}")
    report_lines.append("- Categorical modes:")
    for c, v in modes.items():
        report_lines.append(f"  - {c}: {v}")

    report_lines.append("\n- Outlier caps (IQR lower/upper):")
    for c, (lo, hi) in caps.items():
        report_lines.append(f"  - {c}: ({lo:.4f}, {hi:.4f})")

    (OUT_DIR / "project_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("Project completed.")
    print(f"Best model: {best_model_name}")
    print(f"Artifacts written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
