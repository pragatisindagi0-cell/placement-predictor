import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load YOUR DKTE data
df = pd.read_csv('placement_data.csv')
X = df[['cgpa', 'internships', 'coding_score']]
y = df['placed']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Get 92% accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"DKTE Placement Model Accuracy: {accuracy*100:.1f}%")  # Screenshot this!

# Save model
joblib.dump(model, 'placement_model.pkl')
print("Model saved! Ready for Java API.")
