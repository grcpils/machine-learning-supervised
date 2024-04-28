import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


def analyze_data(data):
    for column in data.columns:
        mean = data[column].mean()
        median = data[column].median()
        mode = data[column].mode()[0]

        print(f'Feature: {column}')
        print(f'Mean: {mean}')
        print(f'Median: {median}')


def preprocess_data(data):
    # Fill NaN values with the mean of the column
    data = data.fillna(data.mean())
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled


def select_best_model(X, y):
    models = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False]
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['friedman_mse', 'squared_error', 'poisson', 'absolute_error'],
                'splitter': ['best', 'random']
            }
        },
        'random_forest': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [10, 50, 100, 200],
                'criterion': ['squared_error', 'absolute_error']
            }
        },
        'svm': {
            'model': SVR(),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
            }
        }
    }

    results = []

    for name, model in models.items():
        grid = GridSearchCV(model['model'], model['params'], cv=5, scoring='r2')
        grid.fit(X, y)
        results.append({
            'model': name,
            'best_score': grid.best_score_,
            'best_params': grid.best_params_
        })

    results.sort(key=lambda x: x['best_score'], reverse=True)

    for result in results:
        print(f"Model: {result['model']}")
        print(f"Best Score (R2): {result['best_score']}")
        print(f"Best Parameters: {result['best_params']}\n")


if __name__ == "__main__":
    data = pd.read_csv('BostonHousing.csv')
    analyze_data(data)

    X = data.drop('medv', axis=1)
    y = data['medv']

    X = preprocess_data(X)
    select_best_model(X, y)
