import argparse
from pathlib import Path
import pickle
import json
import time
import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import TunedThresholdClassifierCV, PredefinedSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support  #, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC  #, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.inspection import permutation_importance

from data_util import preprocess_data


# Global configs
SEED = 42
N_JOBS = -1
SCORING = 'f1'
VERBOSE = 0


### Logistic Regression ###

def _logistic_regression_feature_importance(model, X, y):
    importances = model.coef_[0].tolist()
    ranked_features = dict(zip(X.columns, importances))
    ranked_features = dict(sorted(ranked_features.items(), key=lambda item: abs(item[1]), reverse=True))
    logging.info(f"Logistic Regression Ranked Features: {ranked_features}")
    return ranked_features

def _logistic_regression_model(X_train, y_train, X_val, y_val, class_weight=None):
    params = {
        'penalty': ['l2'],
        'C': [100],  # [10, 100, 1000],
        'solver': ['lbfgs', 'sag'],  # ['lbfgs', 'sag', 'saga'],
        'max_iter': [3000],
        # 'class_weight': [None, 'balanced'],
    }
    model = LogisticRegression(class_weight=class_weight, random_state=SEED, n_jobs=N_JOBS, verbose=VERBOSE)
    
    return _fit(X_train, y_train, X_val, y_val, model, params=params, 
                feature_importance_fn=_logistic_regression_feature_importance)


### Linear SVC ###

def _linear_svc_feature_importance(model, X, y):
    # Aggregate feature importances from each calibrated classifier
    coef_list = []
    for clf in model.calibrated_classifiers_:
        coef_list.append(clf.estimator.coef_[0])
    importances = np.mean(coef_list, axis=0).tolist()
    ranked_features = dict(zip(X.columns, importances))
    ranked_features = dict(sorted(ranked_features.items(), key=lambda item: abs(item[1]), reverse=True))
    logging.info(f"Calibrated LinearSVC Ranked Features: {ranked_features}")
    return ranked_features

def _linear_svc_model(X_train, y_train, X_val, y_val, class_weight=None):
    params = {
        'C': [1, 10],
        # 'class_weight': [None, 'balanced']
    }
    model = LinearSVC(max_iter=5000, class_weight=class_weight, random_state=SEED, verbose=VERBOSE)

    return _fit(X_train, y_train, X_val, y_val, model, params=params, 
                feature_importance_fn=_linear_svc_feature_importance, calibration=True)


### MLP ###

def _mlp_model(X_train, y_train, X_val, y_val):  #, class_weight=None):
    params = {
        'hidden_layer_sizes': [(128, 256, 128)],  # [(128,), (64, 128, 64), (128, 128, 128)], (256, 256)]
        'activation': ['relu'],  #, 'tanh'],
        'solver': ['adam'],
        'alpha': [1e-3],  # 0.0001, 
        'learning_rate': ['constant'],  # 'adaptive'
    }
    model = MLPClassifier(max_iter=10000, early_stopping=True, validation_fraction=0.05, n_iter_no_change=20, 
                          random_state=SEED, verbose=VERBOSE)
    
    return _fit(X_train, y_train, X_val, y_val, model, params=params)


### Random Forest ###

def _tree_based_feature_importance(model, X, y):
    importances = model.feature_importances_.tolist()
    ranked_features = dict(zip(X.columns, importances))
    ranked_features = dict(sorted(ranked_features.items(), key=lambda item: item[1], reverse=True))
    logging.info(f"Tree-based (Gini) Ranked Features: {ranked_features}")
    return ranked_features

def _random_forest_model(X_train, y_train, X_val, y_val, class_weight=None):
    params = {
        'n_estimators': [400],  # [100, 200, 400],
        'max_depth': [None, 18],  # [None, 6, 12, 18],
        'min_samples_split': [5],  # [2, 5, 10],
        'min_samples_leaf': [2],  # [1, 2, 4],
        'max_features': ['sqrt'],  # ['sqrt', 'log2', None],
        'bootstrap': [True],  # , [True, False],
        # 'class_weight': [None, 'balanced'],
    }
    model = RandomForestClassifier(class_weight=class_weight, random_state=SEED, n_jobs=N_JOBS, verbose=VERBOSE)
    
    return _fit(X_train, y_train, X_val, y_val, model, params=params, 
                feature_importance_fn=_tree_based_feature_importance)
    

### XGBoost ###

def _xgboost_model(X_train, y_train, X_val, y_val, class_weight=None):
    scale_pos_weight = class_weight[1] / class_weight[0] if class_weight is not None else None
    params = {
        'n_estimators': [400],  # [100, 200, 400],
        'max_depth': [0, 12],  # [3, 6, 9],
        'learning_rate': [1e-2],  #, 1e-1],  # [0.01, 0.1, 0.3],
        'subsample': [0.8],  # [0.8, 1.0],
        'colsample_bytree': [0.8],  # [0.8, 1.0],
        # 'gamma': [0],  # [0, 0.1],
        # 'reg_alpha': [0],
        # 'reg_lambda': [1],
        # 'tree_method': ['auto'],
        # 'scale_pos_weight': [1, scale_pos_weight],
    }
    model = XGBClassifier(eval_metric='logloss', scale_pos_weight=scale_pos_weight, 
                          random_state=SEED, n_jobs=N_JOBS, verbosity=VERBOSE)
    
    return _fit(X_train, y_train, X_val, y_val, 
                model, params=params, 
                feature_importance_fn=_tree_based_feature_importance)
   

### Stacking Classifier ### 

def _stacking_model(X_train, y_train, X_val, y_val, estimators):  #, class_weight=None):
    model = StackingClassifier(
        estimators=estimators,
        cv='prefit',
        n_jobs=N_JOBS, 
        verbose=VERBOSE
    )
    return _fit(X_val, y_val, X_val, y_val, model)


### Common ###

def _fit(X_train, y_train, X_val, y_val, model, params=None, feature_importance_fn=None, calibration=False):
    best_params = None
    if params is not None:
        # Create a test_fold array where -1 indicates training data and 0 indicates validation data
        X_train_val = np.concatenate([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])
        test_fold = [-1] * len(X_train) + [0] * len(X_val)
        ps = PredefinedSplit(test_fold)
        
        # Hyperparameter tuning
        def n_configs(params):
            if isinstance(params, dict):
                return np.prod([len(v) for v in params.values()])
            elif isinstance(params, list):
                return sum([n_configs(p) for p in params])
            else:
                raise ValueError("Invalid parameter type")
        
        n_iter = 20
        if n_configs(params) > n_iter:
            Search = RandomizedSearchCV
            search_kwargs = dict(n_iter=n_iter, random_state=SEED)
        else:
            Search = GridSearchCV
            search_kwargs = {}
        
        search = Search(
            model,
            params,
            cv=ps, 
            refit=False,
            scoring=SCORING,
            n_jobs=N_JOBS,
            verbose=VERBOSE,
            **search_kwargs
        )
        search.fit(X_train_val, y_train_val)
        
        # Set the best parameters       
        best_params = search.best_params_
        model.set_params(**best_params)
        logging.info(f"Best Parameters: {best_params}")
    
    # Fit the model
    t1 = time.time()
    best_model = model.fit(X_train, y_train)
    # best_model = model
    t2 = time.time()
    training_time = round(t2 - t1, 2)
    
    # Calibration
    if calibration:
        best_model = CalibratedClassifierCV(best_model, cv='prefit', n_jobs=N_JOBS)
        best_model.fit(X_val, y_val)

    # Feature importance
    ranked_features = None
    if feature_importance_fn:
        ranked_features = feature_importance_fn(best_model, X_train, y_train)

    # Tune threshold
    tuned_model = TunedThresholdClassifierCV(best_model, cv='prefit', refit=False, scoring=SCORING, n_jobs=N_JOBS)
    tuned_model.fit(X_val, y_val)
    best_threshold = tuned_model.best_threshold_.item()
    best_score = tuned_model.best_score_.item()
    
    return tuned_model, {
        'best_params': best_params,
        'ranked_features': ranked_features,
        'best_threshold': best_threshold,
        'best_score': best_score, 
        'training_time': training_time
    }


def _evaluate_model(model, X_test, y_test):
    t1 = time.time()
    y_pred = model.predict(X_test)
    t2 = time.time()
    inference_time = round(t2 - t1, 2)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average='binary')

    roc_auc = None
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)

    metrics = {'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc}
    metrics = {k: round(100 * v.item(), 2) if v is not None else None for k, v in metrics.items()}
    metrics['inference_time'] = inference_time
    logging.info(f"Metrics: {metrics}")
    
    return metrics


def _feature_importance(model, X, y):
    # Calculate permutation feature importance
    result = permutation_importance(model, X, y, n_repeats=10, scoring=SCORING, random_state=SEED, n_jobs=N_JOBS)
    importances = result.importances_mean.tolist()
    ranked_pfi = {name: importance for name, importance in zip(X.columns, importances)}
    ranked_pfi = dict(sorted(ranked_pfi.items(), key=lambda item: item[1], reverse=True))
    logging.info(f"PFI Ranked Features: {ranked_pfi}")
    return ranked_pfi


def train_and_evaluate_models(Xy_train, Xy_validation, Xy_test, save_dir, class_weight=None):
    """
    Train and evaluate multiple machine learning models, saving the trained models and their results.
    Parameters:
        Xy_train (tuple): A tuple containing the training features and labels.
        Xy_validation (tuple): A tuple containing the validation features and labels.
        Xy_test (tuple): A tuple containing the testing features and labels.
        save_dir (str): The directory where the models and results will be saved.
        class_weight (dict, optional): Class weights for imbalanced datasets. Defaults to None.
    Returns:
        tuple: A tuple containing the path to the results CSV file and a DataFrame with the results.
    
    The function trains and evaluates the following models:
    - Logistic Regression
    - Random Forest
    - XGBoost
    For each model, it checks if a cached version exists. If so, it loads the model and results from the cache.
    Otherwise, it trains the model, evaluates it, and saves the model and results to the specified directory.
    The results include:
    - Model name
    - Best/tuned hyper-parameters
    - Metrics (precision, recall, F1 score, ROC AUC)
    - Ranked features (based on model-specific feature importance if applicable)
    - Ranked permutation feature importance (PFI)
    - Training and inference runtime
    - Paths to the saved model and results files
    Finally, it saves all results to a CSV file in the specified directory.
    """
    
    def _train_and_evaluate(model_name, train_fn, **train_kwargs):
        model_save_path = Path(save_dir) / f'{model_name}_model.pkl'
        results_save_path = Path(save_dir) / f'{model_name}_results.json'
        if model_save_path.exists() and results_save_path.exists():
            logging.info(f"Loading {model_name} model from cache: {model_save_path}")
            with open(model_save_path, 'rb') as f:
                model = pickle.load(f)
                
            logging.info(f"Loading {model_name} model results from cache: {results_save_path}")
            with open(results_save_path, 'r') as f:
                results = json.load(f)
        else:
            logging.info(f"\nTraining {model_name} Model...")
            model, train_info = train_fn(*Xy_train, *Xy_validation, **train_kwargs)  #, class_weight=class_weight)
            logging.info(f"Evaluating {model_name} Model...")
            metrics = _evaluate_model(model, *Xy_test)
            ranked_pfi = _feature_importance(model, *Xy_test)
            
            logging.info(f"Saving {model_name} model to {model_save_path}")
            with open(model_save_path, 'wb') as f:
                pickle.dump(model, f)
                
            results = {'model': model_name, **train_info, **metrics, 'ranked_pfi': ranked_pfi, 
                       'results_save_path': str(results_save_path), 'model_save_path': str(model_save_path)}
            logging.info(f"Saving {model_name} model results to {results_save_path}")
            with open(results_save_path, 'w') as f:
                json.dump(results, f, indent=4)
                
        return model, results
            
    results = []
    estimators = []
    for model_name, train_fn in [
        ('logistic_regression', _logistic_regression_model),
        ('linear_svc', _linear_svc_model),
        ('mlp', _mlp_model),
        ('random_forest', _random_forest_model),
        ('xgboost', _xgboost_model),
    ]:
        train_kwargs = {'class_weight': class_weight}
        if model_name in ['mlp']:
            train_kwargs = {}
            if class_weight is not None:
                logging.warning(f"Class weights are not supported for {model_name}, skipping.")
                continue
        best_model, results_model = _train_and_evaluate(model_name, train_fn, **train_kwargs)
        results.append(results_model)
        estimators.append((model_name, best_model))
    
    # Stacking Classifier
    best_stacking, results_stacking = _train_and_evaluate('stacking', _stacking_model, estimators=estimators)
    results.append(results_stacking)
    
    # Save all results
    results_path = Path(save_dir) / 'all_results.csv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)
    
    return results_path, results_df
    

def claim_prediction(input_path, test_size=0.1, balance_strategy=None, outlier_strategy=None, outlier_boundary_method=None, save_dir="output"):
    """
    Claim Prediction API: Reads the input data, preprocesses it, trains and evaluates models, and saves the results.
    Args:
        input_path (str): Path to the input CSV file containing the data.
        test_size (float, optional): The proportion of the data to be used for each of the test and validation sets. Defaults to 0.1.
        outlier_strategy (str, optional): Strategy to handle outliers (e.g., 'cap', 'remove'). Defaults to None.
        outlier_boundary_method (str, optional): Method to determine outlier boundaries (e.g., 'IQR', 'percentile'). Defaults to None.
        save_dir (str, optional): Directory to save the output results and logs. Defaults to "output".
    Returns:
        tuple: A tuple containing the path to the results CSV file and a DataFrame with the results.
    """
    
    data_id = f"{Path(input_path).stem}-{test_size}_balance-{balance_strategy}_outliers-{outlier_strategy}-{outlier_boundary_method}_seed-{SEED}"
    save_dir = Path(save_dir) / data_id
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging_path = save_dir / 'all_outputs.log'
    logging.basicConfig(
        level=logging.INFO,  # Set the minimum logging level
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
        datefmt='%Y-%m-%d %H:%M:%S',  # Date format
        handlers=[
            logging.FileHandler(logging_path),  # Use the provided log file path
            logging.StreamHandler()  # Log to console
        ], 
        force=True
    )
    
    # Read data
    df = pd.read_csv(input_path)
    
    # Preprocess data
    Xy_splits = preprocess_data(df, test_size=test_size,
                                balance_strategy=balance_strategy,
                                outlier_strategy=outlier_strategy, outlier_boundary_method=outlier_boundary_method)

    # Train and evaluate models
    y_train_value_counts = Xy_splits['train'][1].value_counts()
    class_weight = {
        0: 1, 
        1: y_train_value_counts[0] / y_train_value_counts[1]
    } if balance_strategy == 'class_weight' else None
    return train_and_evaluate_models(Xy_splits['train'], Xy_splits['validation'], Xy_splits['test'], save_dir, class_weight=class_weight)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Path to input CSV file')
    parser.add_argument('--test_size', type=float, default=0.1, help='Proportion of data to use for test and validation sets')
    parser.add_argument('--balance_strategy', type=str, default=None, help='Class imbalance handling strategy (None, class_weight, SMOTE, ADASYN)')
    parser.add_argument('--outlier_strategy', type=str, default=None, help='Outlier handling strategy (None, cap, remove)')
    parser.add_argument('--outlier_boundary_method', type=str, default=None, help='Outlier boundary method (None, IQR, percentile)')
    parser.add_argument('--save_dir', type=str, default="output", help='Directory to save best models, parameters, results, and logs')
    args = parser.parse_args()
    
    claim_prediction(input_path=args.input_path, test_size=args.test_size,
                     balance_strategy=args.balance_strategy,
                     outlier_strategy=args.outlier_strategy, outlier_boundary_method=args.outlier_boundary_method, 
                     save_dir=args.save_dir)