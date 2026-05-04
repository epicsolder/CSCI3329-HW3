import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------------------------------------------------------
# 1) Load the dataset (space‑separated, no header)
# ---------------------------------------------------------
df = pd.read_csv("your_dataset.txt", sep=" ", header=None, engine="python")

# ---------------------------------------------------------
# 2) Assign column names (A1, A2, A3, ..., A21)
# Your dataset has 21 attributes per row.
# ---------------------------------------------------------
df.columns = [f"A{i}" for i in range(1, len(df.columns) + 1)]

# ---------------------------------------------------------
# 3) Drop rows with missing values
# ---------------------------------------------------------
df = df.dropna()

# ---------------------------------------------------------
# 4) Drop irrelevant columns (none here, but kept for assignment format)
# ---------------------------------------------------------
irrelevant = []   # e.g., ['ID']
df = df.drop(columns=irrelevant, errors='ignore')

# ---------------------------------------------------------
# 5) Identify target and features
# In the German Credit dataset, the LAST column is the class label (A201)
# ---------------------------------------------------------
target_col = df.columns[-1]
y = df[target_col]
X = df.drop(columns=[target_col])

# ---------------------------------------------------------
# 6) Encode categorical features
# Any column containing strings like "A11", "A34", etc. is categorical
# ---------------------------------------------------------
for col in X.columns:
    if X[col].dtype == object:
        X[col] = LabelEncoder().fit_transform(X[col])

# ---------------------------------------------------------
# 7) Encode target if needed
# ---------------------------------------------------------
if y.dtype == object:
    y = LabelEncoder().fit_transform(y)

# ---------------------------------------------------------
# 8) Scale numerical features
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Shape:", X_scaled.shape)
print("Classes:", set(y))
