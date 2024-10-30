import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN


# Global configs
SEED = 42
N_JOBS = -1


def _encode_categoricals(df):
    logging.info("Encoding categorical variables: Gender, Vehicle_Age, and Previous_Vehicle_Damage")
    
    # Gender
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Vehicle_Age
    vehicle_age_mapping = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
    df['Vehicle_Age'] = df['Vehicle_Age'].map(vehicle_age_mapping)

    # Previous_Vehicle_Damage
    df['Previous_Vehicle_Damage'] = df['Previous_Vehicle_Damage'].map({'Yes': 1, 'No': 0})
    
    return df


def _train_test_split(X, y, test_size=0.1, validation_size=0.1):
    logging.info(f"Splitting the data into train, validation, and test sets (test_size={test_size}, validation_size={validation_size})")
    
    n = X.shape[0]
    if isinstance(test_size, float):
        test_size = int(test_size * n)
    if isinstance(validation_size, float):
        validation_size = int(validation_size * n)
    assert test_size + validation_size < n, "Test and validation sizes are too large"

    # Train-test split with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_size, stratify=y_train, random_state=SEED
    )
    
    return {
        'train': (X_train, y_train), 
        'validation': (X_val, y_val),
        'test': (X_test, y_test)
    }
   

def _handle_outliers(Xy_splits, strategy='cap', boundary_method='IQR'):
    logging.info(f"Handling outliers using '{strategy}' strategy and '{boundary_method}' boundary method")
    
    def _outlier_boundaries(X, col, method):
        if method == 'IQR':
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return lower_bound, upper_bound
        elif method == 'percentile':
            lower_bound = X[col].quantile(0.01)
            upper_bound = X[col].quantile(0.99)
            return lower_bound, upper_bound
        else:
            raise ValueError("Please specify the method as 'IQR' or 'percentile'")
        
    for col in ['Age', 'Annual_Premium']:
        # Calculate the boundaries based on the training set
        lower_bound, upper_bound = _outlier_boundaries(Xy_splits['train'][0], col, method=boundary_method)
    
        if strategy == 'cap':
            # Cap the outliers in all splits
            for split in Xy_splits:
                Xy_splits[split][0][col] = Xy_splits[split][0][col].clip(lower=lower_bound, upper=upper_bound)
        elif strategy == 'remove':
            # Remove the outliers only from the training set (Added for comparison purposes, not recommended in practice)
            filter = (Xy_splits['train'][0][col] >= lower_bound) & (Xy_splits['train'][0][col] <= upper_bound)
            Xy_splits['train'] = (Xy_splits['train'][0][filter], Xy_splits['train'][1][filter])
        else:
            raise ValueError("Please specify the strategy as 'cap' or 'remove'")
    
    return Xy_splits


def _scale_numericals(Xy_splits):
    logging.info("Scaling numerical variables: 'Age' and 'Annual_Premium'")
    
    age_scaler = StandardScaler().fit(Xy_splits['train'][0][['Age']])
    for split in Xy_splits:
        Xy_splits[split][0]['Age'] = age_scaler.transform(Xy_splits[split][0][['Age']])
    
    # premium_scaler = RobustScaler().fit(Xy_splits['train'][0][['Annual_Premium']])
    premium_scaler = StandardScaler().fit(Xy_splits['train'][0][['Annual_Premium']])
    for split in Xy_splits:
        Xy_splits[split][0]['Annual_Premium'] = premium_scaler.transform(Xy_splits[split][0][['Annual_Premium']])
    
    return Xy_splits
    

def _handle_class_imbalance(X, y, strategy='SMOTE'):
    logging.info(f"Handling class imbalance using '{strategy}' strategy")
    if strategy == 'SMOTE':
        smote = SMOTE(random_state=SEED, n_jobs=N_JOBS)
        return smote.fit_resample(X, y)
    elif strategy == 'ADASYN':
        adasyn = ADASYN(random_state=SEED, n_jobs=N_JOBS)
        return adasyn.fit_resample(X, y)
    else:
        raise ValueError("Invalid class imbalance handling strategy.")


def preprocess_data(df, test_size=0.1, balance_strategy=None, outlier_strategy=None, outlier_boundary_method=None):
    """
    Preprocesses the input DataFrame: encoding categorical variables, optional handling of outliers, scaling numerical variables, \
        optional handling of class imbalance, and splitting the data into train, validation, and test sets.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data to be preprocessed.
        test_size (float, optional): The proportion of the data to be used for each of the test and validation sets. Default is 0.1.
        balance_strategy (str, optional): The strategy to handle class imbalance. Options are 'SMOTE' and 'ADASYN'. Default is None.
        outlier_strategy (str, optional): The strategy to handle outliers in the 'Annual_Premium' column. Options are 'cap' and 'remove'. Default is None.
        outlier_boundary_method (str, optional): The method to determine the boundary for outliers. Options are 'IQR' and 'percentile'. Default is None.
    Returns:
        dict: A dictionary containing the train, validation, and test splits of the preprocessed data.
    """
    
    def _log_shape(data, prefix=""):
        if isinstance(data, dict):
            info = " | ".join([f"{split.capitalize()}: {X.shape}, {y.shape}" for split, (X, y) in data.items()])
            logging.info(f"{prefix}: {info}")
        elif isinstance(data, tuple):
            X, y = data
            logging.info(f"{prefix}: X: {X.shape}, y: {y.shape}")
        elif isinstance(data, pd.DataFrame):
            logging.info(f"{prefix}: {data.shape}")
        else:
            raise ValueError("Invalid data type")
    
    # Drop irrelevant columns
    df = df.drop(['Claim ID'], axis=1)
    
    # Encode categorical variables
    df = _encode_categoricals(df)
    
    # Features and target variable
    X = df.drop('Response', axis=1)
    y = df['Response']
    _log_shape((X, y), prefix="Initial data")
    
    # Train-test split
    Xy_splits = _train_test_split(X, y, test_size=test_size, validation_size=test_size)

    # Handle outliers in 'Annual_Premium' column based on the training set
    if outlier_strategy:
        Xy_splits = _handle_outliers(Xy_splits, strategy=outlier_strategy, boundary_method=outlier_boundary_method)
    
    # Scale numerical variables based on the training set
    Xy_splits = _scale_numericals(Xy_splits)

    # Handle class imbalance only for the training set
    if balance_strategy in ['SMOTE', 'ADASYN']:
        Xy_splits['train'] = _handle_class_imbalance(*Xy_splits['train'], strategy=balance_strategy)

    _log_shape(Xy_splits, prefix="Final data")
    
    return Xy_splits
