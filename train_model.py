import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load descriptors and descriptor list
desc = pd.read_csv('descriptors_output.csv')
Xlist = list(pd.read_csv('descriptor_list.csv').columns)
X = desc[Xlist]

# Dummy y values (for now)
y = np.random.uniform(5, 9, size=(X.shape[0],))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_train, y_train)
r2 = model.score(X_test, y_test)
r2


y_pred = model.predict(X)
y_pred





# Save model
with open('acetylcholinesterase_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved.")

