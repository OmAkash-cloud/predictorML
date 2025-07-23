import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    'study_hours':     [2, 5, 1, 7, 4, 3, 6, 8, 1, 5],
    'attendance':      [60, 90, 50, 95, 80, 70, 85, 100, 40, 75],
    'previous_score':  [55, 78, 40, 88, 65, 58, 75, 90, 35, 70],
    'sleep_hours':     [5, 7, 4, 8, 6, 6, 7, 8, 3, 6],
    'tutoring':        ['no', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'yes', 'no', 'yes'],
    'pass':            ['fail', 'pass', 'fail', 'pass', 'pass', 'fail', 'pass', 'pass', 'fail', 'pass']
}

df = pd.DataFrame(data)
df['tutoring'] = df['tutoring'].map({'no': 0, 'yes': 1})
df['pass'] = df['pass'].map({'fail': 0, 'pass': 1})

X = df.drop('pass', axis=1)
y = df['pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

new_student = np.array([[1, 40, 35, 4, 0]])
prediction = model.predict(new_student)
print("New Student Prediction:", "PASS ✅" if prediction[0] == 1 else "FAIL ❌")

sns.pairplot(df, hue="pass")
plt.title("Student Data Visualization")
plt.show()
