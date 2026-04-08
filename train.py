import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

#Exercise 1

# Load coffee analysis data from URL
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

# Prepare features and target
X = df[["100g_USD"]]
y = df["rating"]

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Save model as pickle file
with open("model_1.pickle", "wb") as f:
	pickle.dump(model, f)

print("Model trained and saved as model_1.pickle")

#Exercise 2
def roast_category(roast):
	roast_map = {
		"Light": 0,
		"Medium-Light": 1,
		"Medium": 2,
		"Medium-Dark": 3,
		"Dark": 4
	}
	return roast_map.get(roast, None)

# Create roast_cat column
df["roast_cat"] = df["roast"].apply(roast_category)

# Prepare features and target
X2 = df[["100g_USD", "roast_cat"]]
y2 = df["rating"]

# Train Decision Tree Regressor
dtr = DecisionTreeRegressor()
dtr.fit(X2, y2)

# Save model 2
with open("model_2.pickle", "wb") as f:
 pickle.dump(dtr, f)

print("Model 2 trained and saved as model_2.pickle")



# Fill missing desc_3 values
df["desc_3"] = df["desc_3"].fillna("")

# Vectorize desc_3 using TF-IDF
vectorizer = TfidfVectorizer()
X3 = vectorizer.fit_transform(df["desc_3"])
y3 = df["rating"]

# Train linear regression model on text
model_3 = LinearRegression()
model_3.fit(X3, y3)

# Save model 3 and the vectorizer (need both!)
with open("model_3.pickle", "wb") as f:
 pickle.dump((model_3, vectorizer), f)

print("Model 3 trained and saved as model_3.pickle")