import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Excel File
file_path = r"C:\student_performance_project\data\student_data.xlsx"

df = pd.read_excel(file_path)

# Show first 5 rows
print("\nFIRST 5 RECORDS\n")
print(df.head())

# Convert Result column into numbers
le = LabelEncoder()

df["Result"] = le.fit_transform(df["Result"])

# Features and Target
X = df[["Attendance", "Math", "Science", "English", "Study_Hours"]]

y = df["Result"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY =", accuracy)

# Average marks
df["Average"] = (
    df["Math"] + df["Science"] + df["English"]
) / 3

# Grade System
def calculate_grade(avg):

    if avg >= 90:
        return "A"

    elif avg >= 75:
        return "B"

    elif avg >= 50:
        return "C"

    else:
        return "F"

df["Grade"] = df["Average"].apply(calculate_grade)

print("\nSTUDENT GRADES\n")

print(df[["Name", "Average", "Grade"]].head())

# Weak Students
weak_students = df[
    (df["Average"] < 50) |
    (df["Attendance"] < 60)
]

print("\nWEAK STUDENTS\n")

print(
    weak_students[
        ["Name", "Attendance", "Average"]
    ]
)

# Attendance vs Average Performance

plt.figure(figsize=(8,5))

plt.scatter(
    df["Attendance"],
    df["Average"]
)

plt.title("Attendance vs Performance")

plt.xlabel("Attendance")

plt.ylabel("Average Marks")

plt.tight_layout()

plt.savefig(
    r"C:\student_performance_project\output\attendance_vs_performance.png"
)

plt.show()

# Student Ranking

df["Rank"] = (
    df["Average"]
    .rank(ascending=False)
)

print("\nSTUDENT RANKING\n")

print(
    df[
        ["Name", "Average", "Rank"]
    ].head()
)

# Improvement Suggestions

print("\nIMPROVEMENT SUGGESTIONS\n")

for index, row in weak_students.iterrows():

    print(
        row["Name"],
        "- Needs improvement in studies and attendance"
    )

# Top 5 Students
top_students = df.sort_values(by="Average", ascending=False)

print("\nTOP 5 STUDENTS\n")
print(top_students[["Name", "Average"]].head())

print("\nTOP 3 STUDENTS\n")

print(
    top_students[
        ["Name", "Average"]
    ].head(3)
)

# Plot Subject Average
subject_avg = [
    df["Math"].mean(),
    df["Science"].mean(),
    df["English"].mean()
]

subjects = ["Math", "Science", "English"]

plt.figure(figsize=(8,5))

plt.bar(subjects, subject_avg)

plt.title("Average Subject Performance")

plt.xlabel("Subjects")
plt.ylabel("Average Marks")

plt.tight_layout()

# Save Graph
plt.savefig(r"C:\student_performance_project\output\subject_performance.png")

print("\nGraph Saved Successfully!")

# Show Graph
plt.show()