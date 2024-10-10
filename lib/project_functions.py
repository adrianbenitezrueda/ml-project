import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import resample
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load dataset from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    pd.set_option('future.no_silent_downcasting', True)
    df.columns = df.columns.str.strip()
    return df

def preprocess_data(df):
    """
    Preprocess the dataset by handling missing values, encoding categorical variables, and dropping unnecessary columns.

    Parameters:
    df (pd.DataFrame): Original DataFrame.

    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna(0)
    df["BMI Category"] = df["BMI Category"].str.replace("Normal Weight", 'Normal')
    df.drop(columns=['Person ID', 'Occupation'], inplace=True)
    df["BMI Category"] = df["BMI Category"].replace({'Normal': 0, 'Overweight': 1, 'Obese': 2})
    df["Sleep Disorder"] = df["Sleep Disorder"].replace({0: 0, 'Insomnia': 1, 'Sleep Apnea': 2}).astype(int)
    df["Gender"] = df["Gender"].map({'Female': 0, 'Male': 1})
    df.drop(columns=['Blood Pressure'], inplace=True)
    df_encoded = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    return df_encoded

def split_data(df):
    """
    Split the dataset into training and testing sets.

    Parameters:
    df (pd.DataFrame): Preprocessed DataFrame.

    Returns:
    tuple: X_train, X_test, y_train, y_test
    """
    X = df.drop('Stress Level', axis=1)
    y = df['Stress Level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train a Gradient Boosting Regressor model.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training labels.

    Returns:
    GradientBoostingRegressor: Trained model.
    """
    gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gbr.fit(X_train, y_train)
    return gbr

def evaluate_model(gbr, X_test, y_test):
    """
    Evaluate the trained model using RMSE, MAE, and R^2 metrics.

    Parameters:
    gbr (GradientBoostingRegressor): Trained model.
    X_test (pd.DataFrame): Testing features.
    y_test (pd.Series): Testing labels.

    Returns:
    tuple: RMSE, MAE, R^2
    """
    y_pred = gbr.predict(X_test)
    rmse =root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return rmse, mae, r2

def balance_data(X_train, y_train):
    """
    Balance the training data by upsampling minority classes to match the majority class count.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training labels.

    Returns:
    tuple: X_train_balanced, y_train_balanced
    """
    train_data = pd.concat([X_train, y_train], axis=1)
    majority_class_count = train_data['Stress Level'].value_counts().max()
    balanced_classes = []
    for label in train_data['Stress Level'].unique():
        class_samples = train_data[train_data['Stress Level'] == label]
        class_samples_upsampled = resample(class_samples, 
                                           replace=True, 
                                           n_samples=majority_class_count, 
                                           random_state=42)
        balanced_classes.append(class_samples_upsampled)
    train_data_balanced = pd.concat(balanced_classes)
    X_train_balanced = train_data_balanced.drop('Stress Level', axis=1)
    y_train_balanced = train_data_balanced['Stress Level']
    return X_train_balanced, y_train_balanced

def plot_feature_importances(gbr, X_train):
    """
    Plot the top 10 feature importances from the trained model.

    Parameters:
    gbr (GradientBoostingRegressor): Trained model.
    X_train (pd.DataFrame): Training features.
    """
    feature_importances = gbr.feature_importances_
    top_features_idx = np.argsort(feature_importances)[-10:]
    top_features_names = X_train.columns[top_features_idx]
    top_features_importances = feature_importances[top_features_idx]
    plt.figure(figsize=(12, 6))
    plt.barh(top_features_names, top_features_importances)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Top 10 Feature Importances')
    plt.gca().invert_yaxis()
    plt.show()

def plot_correlation_matrix(df):
    """
    Plot the correlation matrix heatmap of the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame to plot the correlation matrix for.
    """
    correlation_matrix = df.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8}, cbar_kws={"shrink": .8}, linewidths=.5, linecolor='black')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

def tune_model(X_train, y_train):
    """
    Tune the Gradient Boosting Regressor model using RandomizedSearchCV.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training labels.

    Returns:
    GradientBoostingRegressor: Best model found by RandomizedSearchCV.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import RandomizedSearchCV
    import numpy as np

    # Definir el estimador
    gbr = GradientBoostingRegressor()

    # Definir el grid de parámetros
    grid = {
        "n_estimators": [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
        "learning_rate": [x for x in np.linspace(start=0.1, stop=1.0, num=10)],
        "max_depth": [int(x) for x in np.linspace(start=1, stop=10, num=5)]
    }

    # Configurar RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=gbr,
        param_distributions=grid,
        n_iter=100,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        error_score='raise'
    )

    # Ajustar el modelo
    random_search.fit(X_train, y_train)

    # Obtener los mejores parámetros y entrenar el modelo con ellos
    best_params = random_search.best_params_
    print("Best parameters found: ", best_params)

    gbr_best = GradientBoostingRegressor(**best_params)
    gbr_best.fit(X_train, y_train)
    return gbr_best


def make_predictions(model, X_test):
    """
    Make predictions using the trained model.

    Parameters:
    model (GradientBoostingRegressor): Trained model.
    X_test (pd.DataFrame): Testing features.

    Returns:
    np.ndarray: Predictions.
    """
    return model.predict(X_test)