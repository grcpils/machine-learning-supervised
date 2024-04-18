import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    Y_train = np.load('Y_train.npy').ravel()
    Y_test = np.load('Y_test.npy').ravel()

    ridge = Ridge(alpha=0.1)
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=1000)

    ridge.fit(X_train, Y_train)
    mlp.fit(X_train, Y_train)

    predictions_ridge = ridge.predict(X_test)
    predictions_mlp = mlp.predict(X_test)

    mse_ridge = np.mean((Y_test - predictions_ridge) ** 2)
    mse_mlp = np.mean((Y_test - predictions_mlp) ** 2)

    print('Ridge MSE:', mse_ridge)
    print('MLP MSE:', mse_mlp)

    r2_ridge = r2_score(Y_test, predictions_ridge)
    print('Ridge R2:', r2_ridge)

    r2_mlp = r2_score(Y_test, predictions_mlp)
    print('MLP R2:', r2_mlp)

    # Define the parameter grid for Ridge
    param_grid_ridge = {
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    # Perform grid search for Ridge
    grid_search_ridge = GridSearchCV(ridge, param_grid_ridge, cv=5, scoring='r2')
    grid_search_ridge.fit(X_train, Y_train)

    # Print the best parameters and R2 score for Ridge
    print('Best parameters for Ridge:', grid_search_ridge.best_params_)
    print('Best R2 score for Ridge:', grid_search_ridge.best_score_)

    # Define the parameter grid for MLP
    param_grid_mlp = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
        'max_iter': [500, 1000, 2000]
    }

    # Perform grid search for MLP
    grid_search_mlp = GridSearchCV(mlp, param_grid_mlp, cv=5, scoring='r2')
    grid_search_mlp.fit(X_train, Y_train)

    # Print the best parameters and R2 score for MLP
    print('Best parameters for MLP:', grid_search_mlp.best_params_)
    print('Best R2 score for MLP:', grid_search_mlp.best_score_)
