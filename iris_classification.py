import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("Iris.csv")

# Dataset Information
print("========== DATASET INFO ==========")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

# Features and Target
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULT ==========")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report")
print(classification_report(y_test, y_pred))


# Graph 1 - Species Count


species_count = df["Species"].value_counts()

plt.figure(figsize=(6,5))
plt.bar(species_count.index, species_count.values)
plt.title("Number of Flowers in Each Species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("species_count.png")
plt.show()


# Graph 2 - Sepal Length Distribution


plt.figure(figsize=(6,5))
plt.hist(df["SepalLengthCm"], bins=12)
plt.title("Sepal Length Distribution")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("sepal_length_distribution.png")
plt.show()


# Graph 3 - Petal Length vs Petal Width

colors = {
    "Iris-setosa": "red",
    "Iris-versicolor": "green",
    "Iris-virginica": "blue"
}

plt.figure(figsize=(7,5))

for species in df["Species"].unique():
    subset = df[df["Species"] == species]
    plt.scatter(
        subset["PetalLengthCm"],
        subset["PetalWidthCm"],
        label=species,
        color=colors[species]
    )

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend()
plt.tight_layout()
plt.savefig("petal_scatter.png")
plt.show()

print("\nProject Completed Successfully!")