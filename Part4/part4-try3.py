from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import numpy as np

if __name__ == "__main__":
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    Y_train = np.load('Y_train.npy').ravel()
    Y_test = np.load('Y_test.npy').ravel()

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    random_forest_regressor = RandomForestRegressor(random_state=0)

    grid_search = GridSearchCV(estimator=random_forest_regressor, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, Y_train)

    print('Best Parameters:', grid_search.best_params_)

    best_model = grid_search.best_estimator_

    best_model.fit(X_train, Y_train)
    predictions = best_model.predict(X_test)
    mse = np.mean((Y_test - predictions) ** 2)
    r2 = r2_score(Y_test, predictions)
    print('Random Forest Regressor MSE:', mse)
    print('Random Forest Regressor R2:', r2)
