import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from typing import Tuple, List, Optional


def remove_unnecessary_columns(df: pd.DataFrame, columns_to_remove: List[str]) -> pd.DataFrame:
    """
    Removes unnecessary columns from the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    columns_to_remove (List[str]): List of column names to be removed.

    Returns:
    pd.DataFrame: DataFrame without the specified columns.
    """
    return df.drop(columns=columns_to_remove, errors="ignore")


def split_data(df: pd.DataFrame, target_col: str, test_size: float, random_state: int) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the dataset into training and validation sets.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    target_col (str): Name of the target column.
    test_size (float): Proportion of data to be used for validation.
    random_state (int): Random seed for reproducibility.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X_train, X_val, y_train, y_val
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def encode_categorical_features(X_train: pd.DataFrame, X_val: pd.DataFrame, categorical_cols: List[str]) -> Tuple[
    pd.DataFrame, pd.DataFrame, OneHotEncoder]:
    """
    Performs One-Hot Encoding on categorical features.

    Parameters:
    X_train (pd.DataFrame): Training data.
    X_val (pd.DataFrame): Validation data.
    categorical_cols (List[str]): List of categorical column names.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder]:
        Transformed training and validation data, along with the encoder instance.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(X_train[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    X_train[encoded_cols] = encoder.transform(X_train[categorical_cols])
    X_val[encoded_cols] = encoder.transform(X_val[categorical_cols])

    X_train = X_train.drop(columns=categorical_cols)
    X_val = X_val.drop(columns=categorical_cols)

    return X_train, X_val, encoder


def scale_numeric_features(X_train: pd.DataFrame, X_val: pd.DataFrame, numeric_cols: List[str]) -> Tuple[
    pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Scales numerical features using Min-Max Scaling.

    Parameters:
    X_train (pd.DataFrame): Training data.
    X_val (pd.DataFrame): Validation data.
    numeric_cols (List[str]): List of numerical column names.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
        Scaled training and validation data, along with the scaler instance.
    """
    scaler = StandardScaler()
    scaler.fit(X_train[numeric_cols])

    X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

    return X_train, X_val, scaler


def define_numerical_cols(df: pd.DataFrame) -> List[str]:
    """
    Identifies numerical columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    List[str]: List of numerical column names.
    """
    return df.select_dtypes(include=np.number).columns.tolist()


def define_categorical_cols(df: pd.DataFrame) -> List[str]:
    """
    Identifies categorical columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    List[str]: List of categorical column names.
    """
    return df.select_dtypes('object').columns.tolist()


def preprocess_data(raw_df: pd.DataFrame, target_col: str, unnecessary_columns: List[str], test_size: float = 0.2,
                    random_state: int = 42, scaler_numeric: bool = True) -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], Optional[MinMaxScaler], OneHotEncoder]:
    """
    Preprocesses the raw dataset by removing unnecessary columns, encoding categorical features, and scaling numerical features.

    Parameters:
    raw_df (pd.DataFrame): Raw input dataset.
    target_col (str): Name of the target column.
    unnecessary_columns (List[str]): List of columns to be removed.
    test_size (float): Proportion of data to be used for validation (default: 0.2).
    random_state (int): Random seed for reproducibility (default: 42).
    scaler_numeric (bool): Whether to scale numerical features (default: True).

    Returns:
    Tuple:
        X_train, y_train, X_val, y_val, input_cols, scaler, encoder
    """

    df = remove_unnecessary_columns(raw_df, unnecessary_columns)
    X_train, X_val, y_train, y_val = split_data(df, target_col, test_size, random_state)

    numeric_cols = define_numerical_cols(X_train)
    categorical_cols = define_categorical_cols(X_train)

    X_train, X_val, encoder = encode_categorical_features(X_train, X_val, categorical_cols)

    scaler = None
    if scaler_numeric:
        X_train, X_val, scaler = scale_numeric_features(X_train, X_val, numeric_cols)

    input_cols = list(X_train.columns)

    return X_train, y_train, X_val, y_val, input_cols, scaler, encoder


def preprocess_new_data(new_df: pd.DataFrame, scaler: Optional[MinMaxScaler],
                        encoder: OneHotEncoder, unnecessary_columns: List[str],
                        ignor_columns: List[str] = None) -> pd.DataFrame:
    """
    Preprocesses new data using previously fitted encoders and scalers.

    Parameters:
    new_df (pd.DataFrame): New input dataset.
    categorical_cols (List[str]): List of categorical columns.
    numeric_cols (List[str]): List of numerical columns.
    scaler (Optional[MinMaxScaler]): Pre-fitted scaler (if available).
    encoder (OneHotEncoder): Pre-fitted encoder.

    Returns:
    pd.DataFrame: Preprocessed dataset.
    """
    df = remove_unnecessary_columns(new_df, unnecessary_columns)

    numeric_cols = define_numerical_cols(df)
    categorical_cols = define_categorical_cols(df)

    if categorical_cols:
        encoded_data = encoder.transform(df[categorical_cols])
        encoded_cols = encoder.get_feature_names_out(categorical_cols)
        df = df.drop(columns=categorical_cols).join(pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index))

    if ignor_columns:
        numeric_cols = [item for item in numeric_cols if item not in ignor_columns]

    if scaler and numeric_cols:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df
