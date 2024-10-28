import logging

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN


# Global configs
SEED = 42
N_JOBS = -1


def _encode_categoricals(df):
    # Gender
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Vehicle_Age
    vehicle_age_mapping = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
    df['Vehicle_Age'] = df['Vehicle_Age'].map(vehicle_age_mapping)

    # Previous_Vehicle_Damage
    df['Previous_Vehicle_Damage'] = df['Previous_Vehicle_Damage'].map({'Yes': 1, 'No': 0})
    
    return df


def _outlier_boundaries(df, col, method):
    if method == 'IQR':
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return lower_bound, upper_bound
    elif method == 'percentile':
        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        return lower_bound, upper_bound
    else:
        raise ValueError("Please specify the method as 'IQR' or 'percentile'")
    

def _handle_outliers(df, col, strategy, boundary_method):
    logging.info(f"Handling outliers for '{col}' using '{strategy}' strategy and '{boundary_method}' method")
    lower_bound, upper_bound = _outlier_boundaries(df, col, method=boundary_method)
    if strategy == 'cap':
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    elif strategy == 'remove':
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    else:
        raise ValueError("Please specify the strategy as 'cap' or 'remove'")
    return df


def _scale_numericals(df):
    # df[['Age', 'Annual_Premium']] = StandardScaler().fit_transform(df[['Age', 'Annual_Premium']])    
    df['Age'] = StandardScaler().fit_transform(df[['Age']])
    df['Annual_Premium'] = RobustScaler().fit_transform(df[['Annual_Premium']])
    return df


def _train_test_split(X, y, test_size=0.1, validation_size=0.1):
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
    
    logging.info(f"Train: {X_train.shape}, {y_train.shape} | " + \
                 f"Validation: {X_val.shape}, {y_val.shape} | " + \
                 f"Test: {X_test.shape}, {y_test.shape}")
    
    return {
        'train': (X_train, y_train), 
        'validation': (X_val, y_val),
        'test': (X_test, y_test)
    }
    

def _handle_class_imbalance(X, y, strategy='SMOTE'):
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
    
    # Drop irrelevant columns
    df = df.drop(['Claim ID'], axis=1)
    
    # Encode categorical variables
    df = _encode_categoricals(df)
    
    # Handle outliers in 'Annual_Premium' column
    if outlier_strategy:
        df = _handle_outliers(df, col='Annual_Premium', strategy=outlier_strategy, boundary_method=outlier_boundary_method)
    
    # Scale numerical variables
    df = _scale_numericals(df)
    
    # Features and target variable
    X = df.drop('Response', axis=1)
    y = df['Response']
    logging.info(f"X: {X.shape}, y: {y.shape}")
    
    if balance_strategy in ['SMOTE', 'ADASYN']:
        X, y = _handle_class_imbalance(X, y, strategy=balance_strategy)
        logging.info(f"X (balanced): {X.shape}, y (balanced): {y.shape}")
    
    # Train-test split
    Xy_splits = _train_test_split(X, y, test_size=test_size, validation_size=test_size)

    return Xy_splits
