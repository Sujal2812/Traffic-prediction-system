import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import seaborn as sns


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
# ── ADD THESE after existing sklearn imports ──────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score
)


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Traffic Prediction System", layout="wide")

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(kernel='rbf')
    }

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)
    train_r2   = r2_score(y_train, train_pred)
    test_r2    = r2_score(y_test,  test_pred)
    cv_scores  = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    return {
        "model":    model,
        "train_r2": round(train_r2, 3),
        "test_r2":  round(test_r2,  3),
        "cv_mean":  round(cv_scores.mean(), 3),
        "cv_std":   round(cv_scores.std(),  3)
    }

def create_sample_data():
    np.random.seed(42)
    n = 500
    hours    = np.random.randint(0, 24, n)
    holidays = np.random.choice([0, 1], n)
    days     = np.random.choice(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], n)
    weathers = np.random.choice(["Clear","Rain","Fog","Snow"], n)

    traffic = (
        50  * np.sin(hours * np.pi / 12) +
        200 * holidays +
        np.where(days == "Fri",  150, 0) +
        np.where(days == "Sat",  100, 0) +
        np.where(weathers == "Rain",  -50, 0) +
        np.where(weathers == "Snow", -100, 0) +
        np.random.normal(0, 50, n)
    )
    traffic = np.maximum(traffic, 0)

    return pd.DataFrame({
        "hour":           hours,
        "holiday":        holidays,
        "day":            days,
        "weather":        weathers,
        "traffic_volume": traffic.astype(int)
    })

def load_and_train_model():
    try:
        df = pd.read_csv("traffic.csv")
    except FileNotFoundError:
        df = create_sample_data()

    le_day     = LabelEncoder()
    le_weather = LabelEncoder()

    df['day_encoded']     = le_day.fit_transform(df['day'])
    df['weather_encoded'] = le_weather.fit_transform(df['weather'])

    X = df[["hour","holiday","day_encoded","weather_encoded"]]
    y = df["traffic_volume"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, le_day, le_weather, df, r2, mae

def create_directories():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data",   exist_ok=True)
 
# ── ADD THESE two functions after create_directories() ────────

def assign_traffic_level(volume):
    """Convert numeric traffic volume into 4 labelled classes"""
    if volume < 100:
        return "🟢 Low"
    elif volume < 250:
        return "🟡 Medium"
    elif volume < 400:
        return "🔴 High"
    else:
        return "🟣 Very High"

def get_classifier_models():
    """Return dict of classification models to compare"""
    return {
        "Logistic Regression":   LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":         RandomForestClassifier(n_estimators=100, random_state=42),
        "Decision Tree":         DecisionTreeClassifier(random_state=42),
        "KNN":                   KNeighborsClassifier(n_neighbors=5),
        "SVM":                   SVC(kernel='rbf', probability=True, random_state=42),
        "Gradient Boosting":     GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
# ✅ FIX 1: keep menu labels consistent — used exactly in every elif below
menu = st.sidebar.selectbox(
    "📌 Navigation",
    ["🏠 Home", "📊 Model Training", "🔮 Prediction", "📈 Model Comparison","🚦 Classification", "ℹ️ About"],
    key="main_navigation"
)

create_directories()

# ==============================
# HOME
# ==============================
if menu == "🏠 Home":
    st.title("🚦 Traffic Prediction Dashboard")
    st.markdown("""
    Welcome to the AI-powered traffic prediction system. This application helps you:

    - **Analyze traffic patterns** based on various factors
    - **Train and compare multiple ML models** for traffic prediction
    - **Make predictions** with confidence intervals
    - **Download prediction reports** for further analysis

    ### Key Features:
    - 📊 Real-time model comparison
    - 🔮 Accurate traffic predictions
    - 📈 Feature importance analysis
    - 🎯 Confidence intervals for predictions
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Models Available", "5+")
    with col2:
        st.metric("Prediction Accuracy", "Up to 95%")
    with col3:
        st.metric("Features Analyzed", "4+")

# ==============================
# MODEL TRAINING
# ==============================
elif menu == "📊 Model Training":
    st.title("📊 Model Training & Evaluation")

    uploaded_file = st.file_uploader("Upload your traffic dataset (CSV)", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    else:
        df = create_sample_data()
        st.info("ℹ️ Using sample dataset. Upload your own CSV for custom training.")

    # ── EDA ──────────────────────────────────────────────
    st.header("🔍 Exploratory Data Analysis (EDA)")

    st.subheader("Dataset Overview")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Dataset Shape:**", df.shape)
    with c2:
        st.write("**Columns:**", list(df.columns))

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Missing Values")
    missing_df = pd.DataFrame({
        'Column':         df.columns,
        'Missing Values': df.isnull().sum().values,
        'Percentage':     (df.isnull().sum().values / len(df) * 100).round(2)
    })
    st.dataframe(missing_df)

    # Histogram
    st.subheader("Numerical Feature Distribution")
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    if num_cols:
        selected_col = st.selectbox("Select column for histogram", num_cols)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[selected_col], bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(f"{selected_col} Distribution", fontsize=14)
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("No numeric columns available.")

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64','float64'])
    if numeric_df.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm",
                    center=0, square=True, linewidths=1, ax=ax)
        ax.set_title("Feature Correlation Matrix", fontsize=14)
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("Not enough numeric columns for heatmap.")

    st.subheader("Dataset Statistics")
    st.dataframe(df.describe())

    with st.expander("Show data preprocessing options"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Categorical columns:**",
                     df.select_dtypes(include=['object']).columns.tolist())
        with c2:
            st.write("**Numerical columns:**",
                     df.select_dtypes(include=['int64','float64']).columns.tolist())

    # Train
    if st.button("🚀 Train Model"):
        with st.spinner("Training in progress..."):
            categorical_cols = df.select_dtypes(include=['object']).columns
            df_encoded = df.copy()
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                label_encoders[col] = le

            X = df_encoded.iloc[:, :-1]
            y = df_encoded.iloc[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2   = r2_score(y_test, y_pred)
            mae  = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("R² Score", f"{r2:.3f}")
            with c2: st.metric("MAE",       f"{mae:.2f}")
            with c3: st.metric("RMSE",      f"{rmse:.2f}")

            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                imp_df = pd.DataFrame({
                    'Feature':    X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(imp_df['Feature'], imp_df['Importance'], color='skyblue')
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importance Analysis')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

            joblib.dump(model,          "models/trained_model.pkl")
            joblib.dump(label_encoders, "models/label_encoders.pkl")
            st.success("✅ Model trained and saved successfully!")

# ==============================
# PREDICTION  ✅ FIX 1 — label matches sidebar exactly
# ==============================
elif menu == "🔮 Prediction":
    st.title("🔮 Predict Traffic")

    if 'prediction_model' not in st.session_state:
        st.session_state.prediction_model    = None
        st.session_state.prediction_encoders = None

    try:
        if os.path.exists("models/trained_model.pkl"):
            st.session_state.prediction_model    = joblib.load("models/trained_model.pkl")
            st.session_state.prediction_encoders = joblib.load("models/label_encoders.pkl")
            st.success("✅ Loaded trained model")
        else:
            model, le_day, le_weather, df, r2, mae = load_and_train_model()
            st.session_state.prediction_model    = model
            st.session_state.prediction_encoders = {'day': le_day, 'weather': le_weather}
            st.info("ℹ️ Using sample model for predictions")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        hour    = st.slider("Hour of Day", 0, 23, 8)
        holiday = st.selectbox("Is Holiday?", [0, 1],
                               format_func=lambda x: "Yes" if x == 1 else "No")
    with col2:
        day     = st.selectbox("Day of Week",
                               ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        weather = st.selectbox("Weather", ["Clear","Rain","Fog","Snow"])

    if st.button("🔮 Predict Traffic", type="primary"):
        try:
            # ── Build input ────────────────────────────────────
            input_df = pd.DataFrame({
                "hour":    [hour],
                "holiday": [holiday],
                "day":     [day],
                "weather": [weather]
            })

            encoders = st.session_state.prediction_encoders

            # Handle both storage layouts (direct vs. nested under column name)
            if "day" in encoders:
                input_df["day"]     = encoders["day"].transform(input_df["day"])
                input_df["weather"] = encoders["weather"].transform(input_df["weather"])
            else:
                # Saved from the Model Training section (keyed by col name)
                for col in ["day", "weather"]:
                    if col in encoders:
                        input_df[col] = encoders[col].transform(
                            input_df[col].astype(str))

            features = ["hour","holiday","day","weather"]
            X_pred   = input_df[features]

            prediction = st.session_state.prediction_model.predict(X_pred)[0]

            # ── Confidence interval via residuals ──────────────
            df_ci = create_sample_data()

            le_day_ci     = LabelEncoder()
            le_weather_ci = LabelEncoder()
            df_ci["day"]     = le_day_ci.fit_transform(df_ci["day"])
            df_ci["weather"] = le_weather_ci.fit_transform(df_ci["weather"])

            X_ci   = df_ci[features]
            y_ci   = df_ci["traffic_volume"]   # ✅ FIX 2 — correct column name

            residuals = y_ci - st.session_state.prediction_model.predict(X_ci)
            std_error = np.std(residuals)

            lower = max(0, prediction - 1.96 * std_error)
            upper = prediction + 1.96 * std_error

            # ── Metrics ────────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Predicted Traffic",    f"{int(prediction):,} vehicles")
            with c2: st.metric("Lower Bound (95% CI)", f"{int(lower):,} vehicles")
            with c3: st.metric("Upper Bound (95% CI)", f"{int(upper):,} vehicles")

            # ── Chart ──────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(["Prediction"], [prediction], color="skyblue", width=0.4)
            ax.errorbar(
                ["Prediction"], [prediction],
                yerr=[[prediction - lower], [upper - prediction]],
                fmt="none", color="red", capsize=10, linewidth=2
            )
            ax.set_ylabel("Traffic Volume (vehicles)")
            ax.set_title("Traffic Prediction with 95% Confidence Interval")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

            # ── Download report ────────────────────────────────
            report_df = pd.DataFrame({
                "Hour":             [hour],
                "Day":              [day],
                "Weather":          [weather],
                "Holiday":          ["Yes" if holiday == 1 else "No"],
                "Predicted Traffic":[int(prediction)],
                "Lower CI (95%)":   [int(lower)],
                "Upper CI (95%)":   [int(upper)]
            })
            csv = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Prediction Report",
                data=csv,
                file_name="traffic_prediction.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# ==============================
# MODEL COMPARISON
# ==============================
elif menu == "📈 Model Comparison":
    st.title("📈 Model Comparison & Analysis")

    df = create_sample_data()

    le_day     = LabelEncoder()
    le_weather = LabelEncoder()
    df['day']     = le_day.fit_transform(df['day'])
    df['weather'] = le_weather.fit_transform(df['weather'])

    X = df[["hour","holiday","day","weather"]]
    y = df["traffic_volume"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    st.subheader("📊 Model Performance Comparison")
    models       = get_models()
    results_list = []
    progress_bar = st.progress(0)

    for i, (name, model) in enumerate(models.items()):
        results = train_and_evaluate(model, X_train, y_train, X_test, y_test)
        y_pred  = results["model"].predict(X_test)
        mae     = mean_absolute_error(y_test, y_pred)
        rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
        results_list.append({
            "Model":      name,
            "Train R²":   results["train_r2"],
            "Test R²":    results["test_r2"],
            "CV R² Mean": results["cv_mean"],
            "CV R² Std":  results["cv_std"],
            "MAE":        round(mae,  2),
            "RMSE":       round(rmse, 2)
        })
        progress_bar.progress((i + 1) / len(models))

    results_df = (pd.DataFrame(results_list)
                    .sort_values("Test R²", ascending=False)
                    .reset_index(drop=True))

    st.dataframe(results_df, use_container_width=True)
    best = results_df.iloc[0]
    st.success(f"🏆 Best Model: **{best['Model']}** (Test R²: {best['Test R²']})")

    tab1, tab2, tab3 = st.tabs(["MAE Comparison","R² Comparison","Bias-Variance"])

    with tab1:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(results_df["Model"], results_df["MAE"],
                      color='skyblue', edgecolor='black')
        ax.set_ylabel("Mean Absolute Error")
        ax.set_title("Model MAE Comparison (Lower is Better)")
        plt.xticks(rotation=45, ha='right')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h,
                    f'{h:.1f}', ha='center', va='bottom')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 6))
        x     = np.arange(len(results_df))
        width = 0.35
        bars1 = ax.bar(x - width/2, results_df["Train R²"], width,
                       label="Train R²", color='blue',  alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, results_df["Test R²"],  width,
                       label="Test R²",  color='green', alpha=0.7, edgecolor='black')
        ax.set_xlabel("Models")
        ax.set_ylabel("R² Score")
        ax.set_title("Train vs Test R² Comparison (Higher is Better)")
        ax.set_xticks(x)
        ax.set_xticklabels(results_df["Model"], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 6))
        bias     = 1 - results_df["Train R²"]
        variance = abs(results_df["Train R²"] - results_df["Test R²"])
        ax.bar(results_df["Model"], bias,     label="Bias (1 - Train R²)",
               color='red',    alpha=0.7, edgecolor='black')
        ax.bar(results_df["Model"], variance, bottom=bias,
               label="Variance (Train-Test Gap)",
               color='orange', alpha=0.7, edgecolor='black')
        ax.set_ylabel("Error Components")
        ax.set_title("Bias-Variance Analysis")
        plt.xticks(rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("🧠 Model Diagnosis")
    c1, c2 = st.columns(2)
    with c1:
        overfit_threshold   = st.slider("Overfitting Threshold (Gap)",    0.05, 0.5, 0.1, 0.05)
    with c2:
        underfit_threshold  = st.slider("Underfitting Threshold (Test R²)", 0.1, 0.8, 0.5, 0.05)

    for _, row in results_df.iterrows():
        train_r2 = row["Train R²"]
        test_r2  = row["Test R²"]
        gap      = round(abs(train_r2 - test_r2), 3)
        with st.expander(f"📊 {row['Model']} Analysis"):
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Train R²", f"{train_r2:.3f}")
            with c2: st.metric("Test R²",  f"{test_r2:.3f}")
            with c3: st.metric("Gap",      f"{gap:.3f}")
            if gap > 0.3:
                st.error("🔴 **Severe Overfitting** — Model is memorising training data")
                st.progress(1.0)
            elif gap > overfit_threshold:
                st.warning("🟡 **Moderate Overfitting** — Consider regularisation or more data")
                st.progress(0.6)
            elif test_r2 < underfit_threshold:
                st.info("🔵 **Underfitting** — Model is too simple, try a more complex model")
                st.progress(0.3)
            else:
                st.success("🟢 **Good Fit** — Model generalises well")
                st.progress(0.1)
 
# ==============================
# CLASSIFICATION SECTION
# ==============================
elif menu == "🚦 Classification":
    st.title("🚦 Traffic Level Classification")
    st.markdown("""
    This section classifies traffic into **4 levels** based on volume:

    | Level | Traffic Volume | Meaning |
    |-------|---------------|---------|
    | 🟢 Low       | < 100 vehicles  | Free-flowing traffic |
    | 🟡 Medium    | 100–249 vehicles | Normal congestion   |
    | 🔴 High      | 250–399 vehicles | Heavy traffic       |
    | 🟣 Very High | 400+ vehicles   | Severe congestion   |
    """)

    # ── Data Preparation ─────────────────────────────────────
    st.header("📦 Dataset Preparation")

    uploaded_clf = st.file_uploader(
        "Upload CSV (optional — uses sample data if not uploaded)",
        type=["csv"],
        key="clf_upload"
    )

    if uploaded_clf is not None:
        df_clf = pd.read_csv(uploaded_clf)
        st.success("✅ File uploaded successfully!")
    else:
        df_clf = create_sample_data()
        st.info("ℹ️ Using generated sample dataset.")

    # Assign class labels
    df_clf["traffic_level"] = df_clf["traffic_volume"].apply(assign_traffic_level)

    # Show class distribution
    st.subheader("📊 Class Distribution")
    col1, col2 = st.columns([1, 1])

    with col1:
        class_counts = df_clf["traffic_level"].value_counts().reset_index()
        class_counts.columns = ["Traffic Level", "Count"]
        st.dataframe(class_counts, use_container_width=True)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ["#2ecc71", "#f1c40f", "#e74c3c", "#9b59b6"]
        bars = ax.bar(
            class_counts["Traffic Level"],
            class_counts["Count"],
            color=colors[:len(class_counts)],
            edgecolor="black"
        )
        ax.set_title("Traffic Level Distribution", fontsize=13)
        ax.set_ylabel("Number of Samples")
        ax.set_xlabel("Traffic Level")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    str(int(h)), ha="center", va="bottom", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Dataset preview
    with st.expander("🔍 Preview labelled dataset"):
        st.dataframe(df_clf.head(20), use_container_width=True)

    st.markdown("---")

    # ── Encode & Split ────────────────────────────────────────
    le_day_c     = LabelEncoder()
    le_weather_c = LabelEncoder()
    le_level_c   = LabelEncoder()

    df_clf["day_enc"]     = le_day_c.fit_transform(df_clf["day"])
    df_clf["weather_enc"] = le_weather_c.fit_transform(df_clf["weather"])
    df_clf["level_enc"]   = le_level_c.fit_transform(df_clf["traffic_level"])

    FEATURES = ["hour", "holiday", "day_enc", "weather_enc"]
    TARGET   = "level_enc"

    X_c = df_clf[FEATURES]
    y_c = df_clf[TARGET]

    test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5,
                          key="clf_test_size") / 100

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_c, y_c, test_size=test_size, random_state=42, stratify=y_c
    )

    st.write(f"🔵 Train samples: **{len(X_train_c)}** | "
             f"🟠 Test samples: **{len(X_test_c)}**")

    # ── Model Selection & Training ────────────────────────────
    st.header("🤖 Train Classification Models")

    all_clf_models = get_classifier_models()
    selected_models = st.multiselect(
        "Select models to train",
        list(all_clf_models.keys()),
        default=["Random Forest", "Decision Tree", "KNN"]
    )

    if st.button("🚀 Train Classifiers", type="primary"):
        if not selected_models:
            st.warning("⚠️ Please select at least one model.")
            st.stop()

        clf_results = []
        progress = st.progress(0)
        status   = st.empty()

        trained_classifiers = {}

        for i, name in enumerate(selected_models):
            status.info(f"⏳ Training **{name}**...")
            clf = all_clf_models[name]
            clf.fit(X_train_c, y_train_c)

            y_train_pred = clf.predict(X_train_c)
            y_test_pred  = clf.predict(X_test_c)

            train_acc = accuracy_score(y_train_c, y_train_pred)
            test_acc  = accuracy_score(y_test_c,  y_test_pred)
            precision = precision_score(y_test_c, y_test_pred,
                                        average="weighted", zero_division=0)
            recall    = recall_score(y_test_c, y_test_pred,
                                     average="weighted", zero_division=0)
            f1        = f1_score(y_test_c, y_test_pred,
                                 average="weighted", zero_division=0)

            clf_results.append({
                "Model":       name,
                "Train Acc":   round(train_acc, 3),
                "Test Acc":    round(test_acc,  3),
                "Precision":   round(precision, 3),
                "Recall":      round(recall,    3),
                "F1-Score":    round(f1,        3)
            })

            trained_classifiers[name] = {"model": clf, "y_pred": y_test_pred}
            progress.progress((i + 1) / len(selected_models))

        status.success("✅ All models trained!")

        results_clf_df = (
            pd.DataFrame(clf_results)
            .sort_values("Test Acc", ascending=False)
            .reset_index(drop=True)
        )

        # ── Results Table ─────────────────────────────────────
        st.subheader("📋 Model Performance Summary")

        def color_accuracy(val):
            if val >= 0.85:
                return "background-color: #d4edda; color: #155724"
            elif val >= 0.70:
                return "background-color: #fff3cd; color: #856404"
            else:
                return "background-color: #f8d7da; color: #721c24"

        styled = results_clf_df.style.applymap(
            color_accuracy,
            subset=["Train Acc", "Test Acc", "Precision", "Recall", "F1-Score"]
        )
        st.dataframe(styled, use_container_width=True)

        best_clf_name = results_clf_df.iloc[0]["Model"]
        best_clf_acc  = results_clf_df.iloc[0]["Test Acc"]
        st.success(f"🏆 Best Classifier: **{best_clf_name}** "
                   f"(Test Accuracy: {best_clf_acc:.1%})")

        # ── Visualisation Tabs ────────────────────────────────
        st.subheader("📈 Visualisations")
        tab1, tab2, tab3, tab4 = st.tabs([
            "Accuracy Comparison",
            "Confusion Matrix",
            "Classification Report",
            "Feature Importance"
        ])

        # ── Tab 1: Accuracy Bar Chart ─────────────────────────
        with tab1:
            fig, ax = plt.subplots(figsize=(10, 6))
            x      = np.arange(len(results_clf_df))
            width  = 0.35

            b1 = ax.bar(x - width/2, results_clf_df["Train Acc"], width,
                        label="Train Accuracy", color="#3498db", alpha=0.85,
                        edgecolor="black")
            b2 = ax.bar(x + width/2, results_clf_df["Test Acc"],  width,
                        label="Test Accuracy",  color="#2ecc71", alpha=0.85,
                        edgecolor="black")

            ax.set_xlabel("Model")
            ax.set_ylabel("Accuracy")
            ax.set_title("Train vs Test Accuracy Comparison", fontsize=13)
            ax.set_xticks(x)
            ax.set_xticklabels(results_clf_df["Model"], rotation=30, ha="right")
            ax.set_ylim(0, 1.15)
            ax.axhline(0.8, color="red", linestyle="--",
                       linewidth=1.2, alpha=0.6, label="80% Threshold")
            ax.legend()
            ax.grid(True, alpha=0.3)

            for bars in [b1, b2]:
                for bar in bars:
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                            f"{h:.2f}", ha="center", va="bottom", fontsize=8)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # F1 / Precision / Recall grouped bar
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            x2     = np.arange(len(results_clf_df))
            w      = 0.25
            ax2.bar(x2 - w,  results_clf_df["Precision"], w,
                    label="Precision", color="#e67e22", alpha=0.85, edgecolor="black")
            ax2.bar(x2,      results_clf_df["Recall"],    w,
                    label="Recall",    color="#9b59b6", alpha=0.85, edgecolor="black")
            ax2.bar(x2 + w,  results_clf_df["F1-Score"],  w,
                    label="F1-Score",  color="#1abc9c", alpha=0.85, edgecolor="black")
            ax2.set_xlabel("Model")
            ax2.set_ylabel("Score")
            ax2.set_title("Precision / Recall / F1-Score Comparison", fontsize=13)
            ax2.set_xticks(x2)
            ax2.set_xticklabels(results_clf_df["Model"], rotation=30, ha="right")
            ax2.set_ylim(0, 1.15)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # ── Tab 2: Confusion Matrix ────────────────────────────
        with tab2:
            cm_model_name = st.selectbox(
                "Select model for confusion matrix",
                list(trained_classifiers.keys()),
                key="cm_select"
            )
            cm_model_data = trained_classifiers[cm_model_name]
            cm = confusion_matrix(y_test_c, cm_model_data["y_pred"])
            class_labels  = le_level_c.classes_

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels,
                yticklabels=class_labels,
                linewidths=0.5, ax=ax
            )
            ax.set_title(f"Confusion Matrix — {cm_model_name}", fontsize=13)
            ax.set_xlabel("Predicted Label", fontsize=11)
            ax.set_ylabel("True Label",      fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Normalised confusion matrix
            st.markdown("**Normalised Confusion Matrix (row %)**")
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm_norm, annot=True, fmt=".2f", cmap="Oranges",
                xticklabels=class_labels,
                yticklabels=class_labels,
                linewidths=0.5, vmin=0, vmax=1, ax=ax2
            )
            ax2.set_title(f"Normalised Confusion Matrix — {cm_model_name}", fontsize=13)
            ax2.set_xlabel("Predicted Label", fontsize=11)
            ax2.set_ylabel("True Label",      fontsize=11)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # ── Tab 3: Classification Report ──────────────────────
        with tab3:
            report_model_name = st.selectbox(
                "Select model for classification report",
                list(trained_classifiers.keys()),
                key="report_select"
            )
            y_rep = trained_classifiers[report_model_name]["y_pred"]
            report_str = classification_report(
                y_test_c, y_rep,
                target_names=le_level_c.classes_,
                zero_division=0
            )
            st.text(f"Classification Report — {report_model_name}")
            st.code(report_str, language="text")

            # Per-class bar chart
            report_dict = classification_report(
                y_test_c, y_rep,
                target_names=le_level_c.classes_,
                output_dict=True,
                zero_division=0
            )
            metrics_per_class = {
                k: v for k, v in report_dict.items()
                if k in le_level_c.classes_
            }
            per_class_df = pd.DataFrame(metrics_per_class).T[
                ["precision", "recall", "f1-score"]
            ]

            fig, ax = plt.subplots(figsize=(9, 5))
            per_class_df.plot(
                kind="bar", ax=ax,
                color=["#3498db","#2ecc71","#e74c3c"],
                edgecolor="black", alpha=0.85
            )
            ax.set_title(f"Per-Class Metrics — {report_model_name}", fontsize=13)
            ax.set_xlabel("Traffic Level")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1.2)
            ax.legend(loc="upper right")
            plt.xticks(rotation=30, ha="right")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── Tab 4: Feature Importance ─────────────────────────
        with tab4:
            fi_model_name = st.selectbox(
                "Select model for feature importance",
                [n for n in trained_classifiers
                 if hasattr(all_clf_models[n], "feature_importances_")],
                key="fi_select"
            )
            if fi_model_name:
                fi_model = trained_classifiers[fi_model_name]["model"]
                feat_names = ["Hour", "Holiday", "Day", "Weather"]
                importances = fi_model.feature_importances_

                fi_df = pd.DataFrame({
                    "Feature":    feat_names,
                    "Importance": importances
                }).sort_values("Importance", ascending=True)

                fig, ax = plt.subplots(figsize=(8, 5))
                colors_fi = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
                ax.barh(fi_df["Feature"], fi_df["Importance"],
                        color=colors_fi, edgecolor="black", alpha=0.85)
                ax.set_xlabel("Importance Score")
                ax.set_title(f"Feature Importance — {fi_model_name}", fontsize=13)
                ax.grid(True, alpha=0.3)

                for i, (val, feat) in enumerate(
                        zip(fi_df["Importance"], fi_df["Feature"])):
                    ax.text(val + 0.002, i, f"{val:.3f}",
                            va="center", fontsize=10)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("ℹ️ Select a tree-based model "
                        "(Random Forest, Decision Tree, Gradient Boosting) "
                        "for feature importance.")

        # ── Save Best Classifier ──────────────────────────────
        best_clf_obj = trained_classifiers[best_clf_name]["model"]
        joblib.dump(best_clf_obj,  "models/best_classifier.pkl")
        joblib.dump(le_level_c,    "models/label_encoder_level.pkl")
        joblib.dump(le_day_c,      "models/label_encoder_day_clf.pkl")
        joblib.dump(le_weather_c,  "models/label_encoder_weather_clf.pkl")

        # ── Live Single-Sample Prediction ────────────────────
        st.markdown("---")
        st.subheader("🎯 Classify a Single Input")

        cp1, cp2 = st.columns(2)
        with cp1:
            c_hour    = st.slider("Hour", 0, 23, 8, key="c_hour")
            c_holiday = st.selectbox("Holiday?", [0, 1], key="c_holiday",
                                     format_func=lambda x: "Yes" if x == 1 else "No")
        with cp2:
            c_day     = st.selectbox("Day",
                                     ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                                     key="c_day")
            c_weather = st.selectbox("Weather",
                                     ["Clear","Rain","Fog","Snow"],
                                     key="c_weather")

        if st.button("🔍 Classify Traffic Level", key="clf_predict_btn"):
            try:
                c_day_enc     = le_day_c.transform([c_day])[0]
                c_weather_enc = le_weather_c.transform([c_weather])[0]

                sample = pd.DataFrame([[c_hour, c_holiday,
                                        c_day_enc, c_weather_enc]],
                                      columns=FEATURES)

                pred_enc   = best_clf_obj.predict(sample)[0]
                pred_label = le_level_c.inverse_transform([pred_enc])[0]

                # Probability bar if supported
                level_colors = {
                    "🟢 Low":       "#2ecc71",
                    "🟡 Medium":    "#f1c40f",
                    "🔴 High":      "#e74c3c",
                    "🟣 Very High": "#9b59b6"
                }
                bg = level_colors.get(pred_label, "#3498db")

                st.markdown(
                    f"""
                    <div style="background-color:{bg};padding:20px;
                                border-radius:12px;text-align:center;
                                color:white;font-size:22px;font-weight:bold;">
                        Predicted Traffic Level: {pred_label}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if hasattr(best_clf_obj, "predict_proba"):
                    proba = best_clf_obj.predict_proba(sample)[0]
                    proba_df = pd.DataFrame({
                        "Traffic Level": le_level_c.classes_,
                        "Probability":   proba
                    }).sort_values("Probability", ascending=True)

                    fig, ax = plt.subplots(figsize=(8, 4))
                    bar_colors = [level_colors.get(l, "#3498db")
                                  for l in proba_df["Traffic Level"]]
                    ax.barh(proba_df["Traffic Level"],
                            proba_df["Probability"],
                            color=bar_colors, edgecolor="black", alpha=0.85)
                    ax.set_xlabel("Probability")
                    ax.set_title("Prediction Probability per Class", fontsize=12)
                    ax.set_xlim(0, 1)
                    ax.axvline(0.5, color="black", linestyle="--",
                               linewidth=1, alpha=0.5)
                    ax.grid(True, alpha=0.3)
                    for i, (p, _) in enumerate(
                            zip(proba_df["Probability"],
                                proba_df["Traffic Level"])):
                        ax.text(p + 0.01, i, f"{p:.2%}",
                                va="center", fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

        # ── Download Classification Report ────────────────────
        st.markdown("---")
        st.subheader("📥 Export Results")

        export_df = pd.DataFrame({
            "Model":     results_clf_df["Model"],
            "Test Acc":  results_clf_df["Test Acc"],
            "Precision": results_clf_df["Precision"],
            "Recall":    results_clf_df["Recall"],
            "F1-Score":  results_clf_df["F1-Score"]
        })
        csv_clf = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Classification Results",
            data=csv_clf,
            file_name="classification_results.csv",
            mime="text/csv"
        )

# ==============================
# ABOUT
# ==============================
elif menu == "ℹ️ About":
    st.title("ℹ️ About This Project")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ## Traffic Prediction System

        This application demonstrates the power of machine learning in predicting traffic patterns.

        ### 🎯 Features
        - **Multiple ML Models**: Compare different regression algorithms
        - **Real-time Predictions**: Get instant traffic predictions
        - **Confidence Intervals**: Understand prediction uncertainty
        - **Model Analysis**: Bias-variance trade-off visualisation
        - **Export Reports**: Download predictions for further analysis

        ### 🛠️ Technologies Used
        - **Streamlit**: Interactive web application framework
        - **Scikit-learn**: Machine learning library
        - **Pandas**: Data manipulation and analysis
        - **Matplotlib / Seaborn**: Data visualisation
        - **Joblib**: Model persistence

        ### 📊 How It Works
        1. **Data Collection**: Historical traffic data with time, weather & holiday features
        2. **Model Training**: Multiple algorithms learn patterns from the data
        3. **Evaluation**: Models compared using R², MAE, RMSE & cross-validation
        4. **Prediction**: Best model used for real-time predictions with CI
        """)
    with col2:
        st.image(
            "https://www.streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png",
            width=200
        )
        st.markdown("---")
        st.markdown("### 📊 System Status")
        st.success("🟢 Online")
        st.info(f"📅 Last Update: {pd.Timestamp.now().strftime('%Y-%m-%d')}")

    st.markdown("---")
    st.markdown("### 🚀 Future Enhancements")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("✅ Real-time traffic data")
    with c2: st.markdown("✅ Deep learning models")
    with c3: st.markdown("✅ Weather API integration")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown("✅ Interactive maps")
    with c2: st.markdown("✅ Time series analysis")
    with c3: st.markdown("✅ Mobile app support")

    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("Built with ❤️ using Streamlit and Machine Learning")
    st.markdown("© 2024 Traffic Prediction System")

# ── Footer ────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")
st.sidebar.success("🟢 Online")
st.sidebar.markdown("---")
st.sidebar.markdown("© 2024 Traffic Prediction System")
