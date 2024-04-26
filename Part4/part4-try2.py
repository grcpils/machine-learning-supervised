import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, neural_network, svm, ensemble
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import VotingRegressor
import xgboost as xgb
import lightgbm as lgb

if __name__ == "__main__":
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    Y_train = np.load('Y_train.npy').ravel()
    Y_test = np.load('Y_test.npy').ravel()

    scaler = StandardScaler()

    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge_params = {'alpha': [0.1, 1.0, 10.0]}
    lasso_params = {'alpha': [0.1, 1.0, 10.0]}
    mlp_params = {'hidden_layer_sizes': [(100, 50, 25), (50, 25, 12)]}
    svr_params = {'kernel': ['linear', 'rbf']}
    adaboost_params = {'n_estimators': [50, 100, 200]}
    xgb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    lgb_params = {'max_depth': [10, 15, 20], 'min_data_in_leaf': [5, 10, 15], 'num_leaves': [50, 100, 200],
                  'learning_rate': [0.01, 0.1, 0.2]}

    models_params = {
        ('Ridge', linear_model.Ridge()): ridge_params,
        ('Lasso', linear_model.Lasso()): lasso_params,
        ('MLPRegressor', neural_network.MLPRegressor(max_iter=1000)): mlp_params,
        ('SVR', svm.SVR()): svr_params,
        ('AdaBoostRegressor', ensemble.AdaBoostRegressor()): adaboost_params,
        ('XGBoost', xgb.XGBRegressor()): xgb_params,
        ('LightGBM', lgb.LGBMRegressor()): lgb_params
    }

    best_models = []

    for (name, model), params in models_params.items():
        grid = GridSearchCV(model, params, cv=5)
        grid.fit(X_train_scaled, Y_train)
        best_models.append((name, grid.best_estimator_))

    stacking_regressor = StackingRegressor(estimators=best_models, final_estimator=linear_model.Ridge())

    stacking_regressor.fit(X_train_scaled, Y_train)
    predictions = stacking_regressor.predict(X_test_scaled)
    mse = np.mean((Y_test - predictions) ** 2)
    r2 = r2_score(Y_test, predictions)
    print('Stacking Regressor MSE:', mse)
    print('Stacking Regressor R2:', r2)

    voting_regressor = VotingRegressor(estimators=best_models)

    voting_regressor.fit(X_train_scaled, Y_train)
    predictions = voting_regressor.predict(X_test_scaled)
    mse = np.mean((Y_test - predictions) ** 2)
    r2 = r2_score(Y_test, predictions)
    print('Voting Regressor MSE:', mse)
    print('Voting Regressor R2:', r2)
