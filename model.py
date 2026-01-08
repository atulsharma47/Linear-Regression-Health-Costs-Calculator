import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def train_model():
    # Load dataset
    data = pd.read_csv("insurance.csv")

    # ---------- Feature Engineering ----------
    # One-hot encoding categorical variables
    data = pd.get_dummies(
        data,
        columns=['sex', 'smoker', 'region'],
        drop_first=True
    )

    # Separate features and target
    X = data.drop('charges', axis=1)
    y = data['charges']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model accuracy
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return model, X.columns, r2
