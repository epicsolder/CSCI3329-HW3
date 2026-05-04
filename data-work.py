import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# ---------------------------------------------------------
# 1) Load file as raw text lines
# ---------------------------------------------------------
with open("german-numeric.csv", "r") as f:
    lines = f.read().strip().split("\n")

# Remove header row
lines = lines[1:]

# ---------------------------------------------------------
# 2) Split each line manually by comma
# ---------------------------------------------------------
rows = []
for line in lines:
    parts = line.split(",")
    parts = parts[1:-1]   # remove empty first + last
    rows.append(parts)

# ---------------------------------------------------------
# 3) Convert to DataFrame
# ---------------------------------------------------------
df = pd.DataFrame(rows)

# ---------------------------------------------------------
# 4) Rename columns automatically
# ---------------------------------------------------------
df.columns = [f"Column{i}" for i in range(1, df.shape[1] + 1)]

print("Loaded shape:", df.shape)
print(df.head())

# ---------------------------------------------------------
# 5) Convert to numeric and drop missing
# ---------------------------------------------------------
df = df.apply(pd.to_numeric, errors="coerce").dropna()

# ---------------------------------------------------------
# 6) Target = last column
# ---------------------------------------------------------
target_col = df.columns[-1]
y = df[target_col]
X = df.drop(columns=[target_col])

print("Target column:", target_col)

# ---------------------------------------------------------
# 7) Encode features
# ---------------------------------------------------------
for col in X.columns:
    X[col] = LabelEncoder().fit_transform(X[col])

y = LabelEncoder().fit_transform(y)

# ---------------------------------------------------------
# 8) Scale features
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Preprocessing complete.")
print("Shape:", X_scaled.shape)
print("Classes:", set(y))

# ---------------------------------------------------------
# 9) Train/Test Split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 10) Models
# ---------------------------------------------------------
models = {
    'Linear Classifier': Perceptron(max_iter=1000, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Gaussian NB': GaussianNB(),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64,),
                                    max_iter=500, random_state=42),
}

# ---------------------------------------------------------
# 11) Train + Evaluate
# ---------------------------------------------------------
results = {}

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))

# ---------------------------------------------------------
# 12) Summary
# ---------------------------------------------------------
print("\n=== Accuracy Summary ===")
for name, acc in results.items():
    print(f"{name:20s}  {acc:.4f}")
