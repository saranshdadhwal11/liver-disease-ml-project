# liver-disease-ml-project
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)

# =========================
# 1. Load and preprocess data
# =========================

df = pd.read_excel("C:\\Users\\Ishu\\Desktop\\bml.xlsx")

# Encode categorical features
categorical_features = ['Gender', 'Alcohol_Intake', 'Physical_Activity', 'Diabetes', 'Hypertension']
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Define features and target
y = df['Liver_Disease']
X = df.drop('Liver_Disease', axis=1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize numeric features
numeric_features = ['Age', 'BMI', 'ALT', 'AST', 'GGT', 'Bilirubin', 'Albumin', 'Platelets']
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_train_scaled.loc[:, numeric_features] = scaler.fit_transform(X_train[numeric_features])

X_test_scaled = X_test.copy()
X_test_scaled.loc[:, numeric_features] = scaler.transform(X_test[numeric_features])

# =========================
# 2. Initialize models
# =========================

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
}

# =========================
# 3. Train, predict, evaluate
# =========================

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print(f"\nModel: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Feature importance (Random Forest only)
    if name == 'Random Forest':
        importances = model.feature_importances_
        feature_names = X.columns
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=feature_names)
        plt.title('Feature Importance - Random Forest')
        plt.show()

# =========================
# 4. Compare model accuracy visually
# =========================

accuracy_scores = {name: accuracy_score(y_test, model.predict(X_test_scaled)) for name, model in models.items()}
plt.figure(figsize=(8, 5))
sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()))
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

# =========================
# 5. Dataset overview
# =========================

print(df.info())
print(df.describe())

# Missing values check
print("\nMissing Values:\n", df.isnull().sum())

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Target distribution (bar chart)
sns.countplot(x='Liver_Disease', data=df)
plt.title("Target Class Distribution")
plt.show()

# =========================
# Target Class Pie Chart (ADDED)
# =========================
plt.figure(figsize=(6, 6))
class_counts = df['Liver_Disease'].value_counts()
plt.pie(
    class_counts,
    labels=["No Liver Disease" if x == 0 else "Liver Disease" for x in class_counts.index],
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette("pastel"),
    shadow=True
)
plt.title("Liver Disease Distribution (Pie Chart)")
plt.show()

# Example feature distribution
sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution")
plt.show()

# =========================
# 6. Cross-validation and ROC Curves
# =========================

for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"{name} CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ROC Curves
plt.figure(figsize=(8, 6))
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    RocCurveDisplay.from_estimator(model, X_test_scaled, y_test, name=name)
plt.title("ROC Curves for All Models")
plt.show()

# =========================
# 7. SHAP Explanation (Fixed)
# =========================

rf_model = models['Random Forest']

explainer = shap.TreeExplainer(rf_model, data=X_train_scaled, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test_scaled, check_additivity=False)
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)

if isinstance(shap_values, list):
    shap_values_to_plot = shap_values[1]
else:
    shap_values_to_plot = shap_values

print("SHAP values shape:", shap_values_to_plot.shape)
print("X_test shape:", X_test_scaled.shape)

shap.summary_plot(shap_values_to_plot, X_test_scaled, plot_type="bar")
shap.summary_plot(shap_values_to_plot, X_test_scaled)

# =========================
# 8. Confusion Matrices
# =========================

for name, model in models.items():
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, cmap='Blues')
    disp.ax_.set_title(f"Confusion Matrix - {name}")
    plt.show()

# =========================
# 9. Pairplot and Results Export
# =========================

sns.pairplot(df, hue="Liver_Disease", vars=['Age', 'BMI', 'ALT', 'AST', 'Bilirubin'])
plt.show()

results = []
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': acc})

results_df = pd.DataFrame(results)
print(results_df)

# Save results
results_df.to_csv("model_performance_summary.csv", index=False)
print("\nResults saved to 'model_performance_summary.csv'")
