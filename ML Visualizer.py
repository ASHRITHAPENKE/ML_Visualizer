import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, silhouette_score, davies_bouldin_score
)
from sklearn.datasets import load_iris, load_digits, fetch_california_housing, make_moons, make_circles
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px

# 🚀 Streamlit App Setup
st.set_page_config(page_title="ML Algorithm Visualizer", layout="centered")
st.title("💻 Machine Learning Algorithm Visualizer")

# 📂 Dataset Selection
dataset_choice = st.sidebar.selectbox("📊 Select Dataset", ["Iris", "California Housing", "Digits", "Custom Upload"])
if dataset_choice == "Iris":
    data = load_iris(as_frame=True)
    df = data.frame
elif dataset_choice == "California Housing":
    data = fetch_california_housing(as_frame=True)
    df = data.frame
elif dataset_choice == "Digits":
    data = load_digits(as_frame=True)
    df = data.frame
elif dataset_choice == "Custom Upload":
    uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    else:
        st.warning("Upload a dataset to proceed.")
        st.stop()

st.write("## 📄 Dataset Preview", df.head())
target_col = st.sidebar.selectbox("🎯 Select Target Variable", df.columns)
feature_cols = st.sidebar.multiselect("📈 Select Feature Columns", df.columns.drop(target_col), default=list(df.columns.drop(target_col))[:3])

# 📊 Task Selection
task_type = st.sidebar.radio("📝 Select Task Type", ["Regression", "Classification", "Clustering", "Dimensionality Reduction"])

# 🧮 Regression
if task_type == "Regression":
    model_name = st.sidebar.selectbox("Select Regressor", ["Linear Regression", "Polynomial Regression", "Ridge", "Lasso", "Decision Tree"])
    test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target_col], test_size=test_size, random_state=42)

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Polynomial Regression":
        from sklearn.preprocessing import PolynomialFeatures
        degree = st.sidebar.slider("Polynomial Degree", 2, 5, 2)
        poly = PolynomialFeatures(degree=degree)
        X_train, X_test = poly.fit_transform(X_train), poly.transform(X_test)
        model = LinearRegression()
    elif model_name == "Ridge":
        alpha = st.sidebar.slider("Ridge Alpha", 0.1, 10.0, 1.0)
        model = Ridge(alpha=alpha)
    elif model_name == "Lasso":
        alpha = st.sidebar.slider("Lasso Alpha", 0.1, 10.0, 1.0)
        model = Lasso(alpha=alpha)
    elif model_name == "Decision Tree":
        max_depth = st.sidebar.slider("Max Depth", 2, 10, 3)
        model = DecisionTreeRegressor(max_depth=max_depth)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    st.write(f"**MSE:** {mean_squared_error(y_test, preds):.4f} | **R²:** {r2_score(y_test, preds):.4f}")
    fig, ax = plt.subplots()
    ax.scatter(y_test, preds, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual"), ax.set_ylabel("Predicted"), ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

# 🏷️ Classification
elif task_type == "Classification":
    model_name = st.sidebar.selectbox("Select Classifier", ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN"])
    test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100

    # 🔑 Handle Continuous Target for Classification
    if df[target_col].dtype in ["float64", "float32"]:
        st.warning("⚠️ Continuous target detected. Using median split for classification.")
        df["discrete_target"] = pd.qcut(df[target_col], q=2, labels=[0, 1])
        target_for_model = "discrete_target"
    else:
        target_for_model = target_col

    X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target_for_model], test_size=test_size, random_state=42)

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=500)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=st.sidebar.slider("Max Depth", 2, 10, 3))
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=st.sidebar.slider("Trees", 10, 200, 100))
    elif model_name == "SVM":
        model = SVC(probability=True, C=st.sidebar.slider("C", 0.1, 10.0, 1.0))
    elif model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=st.sidebar.slider("Neighbors (k)", 1, 15, 5))

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f} | **Precision:** {precision_score(y_test, y_pred, average='weighted'):.4f} | **Recall:** {recall_score(y_test, y_pred, average='weighted'):.4f} | **F1:** {f1_score(y_test, y_pred, average='weighted'):.4f}")

    # 📉 Confusion Matrix
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted"), ax.set_ylabel("Actual"), ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# 🔎 Clustering
elif task_type == "Clustering":
    model_name = st.sidebar.selectbox("Select Clustering Algorithm", ["K-Means", "DBSCAN"])
    X = StandardScaler().fit_transform(df[feature_cols])

    if model_name == "K-Means":
        k = st.sidebar.slider("Number of Clusters", 2, 10, 3)
        model = KMeans(n_clusters=k, random_state=42)
    elif model_name == "DBSCAN":
        eps = st.sidebar.slider("EPS", 0.1, 2.0, 0.5)
        model = DBSCAN(eps=eps, min_samples=5)

    clusters = model.fit_predict(X)
    st.write(f"**Silhouette Score:** {silhouette_score(X, clusters):.4f} | **Davies-Bouldin Index:** {davies_bouldin_score(X, clusters):.4f}")
    df["Cluster"] = clusters
    fig = px.scatter_matrix(df, dimensions=feature_cols, color="Cluster", title="Cluster Visualization")
    st.plotly_chart(fig, use_container_width=True)

# 🧭 Dimensionality Reduction
elif task_type == "Dimensionality Reduction":
    method = st.sidebar.selectbox("Select Method", ["PCA", "t-SNE"])
    components = st.sidebar.slider("Components", 2, 3, 2)
    X = StandardScaler().fit_transform(df[feature_cols])

    if method == "PCA":
        reducer = PCA(n_components=components)
    else:
        reducer = TSNE(n_components=components, perplexity=st.sidebar.slider("Perplexity", 5, 50, 30))

    reduced = reducer.fit_transform(X)
    st.write("### Reduced Dimensions Visualization")
    if components == 2:
        fig = px.scatter(x=reduced[:, 0], y=reduced[:, 1], title=f"{method} Visualization")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.scatter_3d(x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2], title=f"{method} 3D Visualization")
        st.plotly_chart(fig, use_container_width=True)
