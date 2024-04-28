import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, neural_network, svm, ensemble
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb

if __name__ == "__main__":
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    Y_train = np.load('Y_train.npy').ravel()
    Y_test = np.load('Y_test.npy').ravel()

    scaler = StandardScaler()

    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        'Ridge': linear_model.Ridge(alpha=0.1),
        'Lasso': linear_model.Lasso(alpha=0.1),
        'MLPRegressor': neural_network.MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=1000),
        'SVR': svm.SVR(kernel='linear'),
        'AdaBoostRegressor': ensemble.AdaBoostRegressor(),
        'XGBoost': xgb.XGBRegressor(),
        'LightGBM': lgb.LGBMRegressor()
    }

    param_grids = {
        'Ridge': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
        'Lasso': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
        'MLPRegressor': {'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)], 'max_iter': [500, 1000, 2000]},
        'SVR': {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': ['scale', 'auto']},
        'AdaBoostRegressor': {'n_estimators': [50, 100, 200], 'learning_rate': [0.001, 0.01, 0.1, 1]},
        'XGBoost': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]},
        'LightGBM': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
    }

    r2_scores = {}

    for name, model in models.items():
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        mse = np.mean((Y_test - predictions) ** 2)
        r2 = r2_score(Y_test, predictions)
        print(f'{name} MSE:', mse)
        print(f'{name} R2:', r2)

        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='r2')
        grid_search.fit(X_train, Y_train)
        print(f'Best parameters for {name}:', grid_search.best_params_)
        print(f'Best R2 score for {name}:', grid_search.best_score_)

        r2_scores[name] = grid_search.best_score_

    sorted_r2_scores = sorted(r2_scores.items(), key=lambda item: item[1], reverse=True)

    print("\nModels ranked by R2 score:")
    for i, (name, score) in enumerate(sorted_r2_scores, start=1):
        print(f"{i}. {name}: {score}")
