import optuna
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def objective(trial, X, y):
    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 10, 200, step=5),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
    }


    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.05,
        shuffle=False
    )

    model = LGBMRegressor(**param_grid)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        verbose=False,
        early_stopping_rounds=100,
    )
    preds = model.predict(X_val)
    return mean_absolute_error(y_val, preds)

def tune(X, y):
    study = optuna.create_study(direction="minimize", study_name="LGBM Regressor")
    func = lambda trial: objective(trial, X.values, y)
    study.optimize(func, n_trials=20)
    return study