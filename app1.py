import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Fetch COVID-19 data foimport requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Fetch COVID-19 data for the USA
url = "https://disease.sh/v3/covid-19/countries/usa"
r = requests.get(url)
data = r.json()

# Extract relevant fields
covid_data = {
    "cases": data["cases"],
    "todayCases": data["todayCases"],
    "deaths": data["deaths"],
    "todayDeaths": data["todayDeaths"],
    "recovered": data["recovered"],
    "active": data["active"],
    "critical": data["critical"],
    "casesPerMillion": data["casesPerOneMillion"],
    "deathsPerMillion": data["deathsPerOneMillion"],
}

# Convert to Pandas DataFrame
df = pd.DataFrame([covid_data])
print(df)

# Plot COVID-19 data
labels = ["Total Cases", "Active Cases", "Recovered", "Deaths"]
values = [data["cases"], data["active"], data["recovered"], data["deaths"]]

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("COVID-19 Data for USA")
plt.show()

# Generate random historical data
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Last 30 days cases
historical_deaths = np.random.randint(500, 2000, size=30)

df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

print(df_historical.head())

# Prepare data for SVM Regression
X = df_historical[["day"]]
y = df_historical["cases"]

# Scale the data (SVM is sensitive to feature scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVM Regression model
svm_model = SVR(kernel='rbf')  # You can change the kernel to 'linear', 'poly', etc.
svm_model.fit(X_train, y_train)

# Predict next day's cases
next_day = np.array([[31]])
next_day_scaled = scaler.transform(next_day)
predicted_cases = svm_model.predict(next_day_scaled)
print(f"Predicted cases for Day 31: {int(predicted_cases[0])}")

# Streamlit App
st.title("COVID-19 Cases Prediction in USA")
st.write("Predicting COVID-19 cases for the next day based on historical data.")

# User Input
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

if st.button("Predict"):
    day_input_scaled = scaler.transform([[day_input]])
    prediction = svm_model.predict(day_input_scaled)
    st.write(f"Predicted cases for day {day_input}: {int(prediction[0])}")r the USA
url = "https://disease.sh/v3/covid-19/countries/usa"
r = requests.get(url)
data = r.json()

# Extract relevant fields
covid_data = {
    "cases": data["cases"],
    "todayCases": data["todayCases"],
    "deaths": data["deaths"],
    "todayDeaths": data["todayDeaths"],
    "recovered": data["recovered"],
    "active": data["active"],
    "critical": data["critical"],
    "casesPerMillion": data["casesPerOneMillion"],
    "deathsPerMillion": data["deathsPerOneMillion"],
}

# Convert to Pandas DataFrame
df = pd.DataFrame([covid_data])
print(df)

# Plot COVID-19 data
labels = ["Total Cases", "Active Cases", "Recovered", "Deaths"]
values = [data["cases"], data["active"], data["recovered"], data["deaths"]]

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=['blue', 'orange', 'green', 'red'])
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("COVID-19 Data for USA")
plt.show()

# Generate random historical data
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Last 30 days cases
historical_deaths = np.random.randint(500, 2000, size=30)

df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

print(df_historical.head())

# Prepare data for SVM Regression
X = df_historical[["day"]]
y = df_historical["cases"]

# Scale the data (SVM is sensitive to feature scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVM Regression model
svm_model = SVR(kernel='rbf')  # You can change the kernel to 'linear', 'poly', etc.
svm_model.fit(X_train, y_train)

# Predict next day's cases
next_day = np.array([[31]])
next_day_scaled = scaler.transform(next_day)
predicted_cases = svm_model.predict(next_day_scaled)
print(f"Predicted cases for Day 31: {int(predicted_cases[0])}")

# Streamlit App
st.title("COVID-19 Cases Prediction in USA")
st.write("Predicting COVID-19 cases for the next day based on historical data.")

# User Input
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

if st.button("Predict"):
    day_input_scaled = scaler.transform([[day_input]])
    prediction = svm_model.predict(day_input_scaled)
    st.write(f"Predicted cases for day {day_input}: {int(prediction[0])}")
