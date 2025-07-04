import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv('winequality-red.csv', sep=';')

X = data.drop('quality', axis=1)
y = data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))

print("✅ Huấn luyện xong, model đã lưu thành model.pkl")
s