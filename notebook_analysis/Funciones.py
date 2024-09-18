import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.offline as pyo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam, SGD  # Import other optimizers as needed
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from tabulate import tabulate
from sklearn.model_selection import KFold
import os
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.offline as pyo
from Funciones import *
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, recall_score, precision_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam, SGD  # Import other optimizers as needed
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tabulate import tabulate
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
import seaborn as sns
import keras
from sklearn.model_selection import KFold
def plot_sensor_comparison_aligned_four_ids(df, engine_id1, engine_id2, engine_id3, engine_id4, cols, sequence_length=50):
    """
    Plots sensor data for four specific engine IDs on the same plot, aligned from the beginning of the data.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    engine_id1 (int): The first engine ID to filter data for.
    engine_id2 (int): The second engine ID to filter data for.
    engine_id3 (int): The third engine ID to filter data for.
    engine_id4 (int): The fourth engine ID to filter data for.
    cols (list): List of column names for the sensors to plot.
    sequence_length (int): The window size in cycles to use for plotting.
    """
    def get_window_data_for_engine(engine_id):
        # Filter data for the given engine ID
        engine_data = df[df['id'] == engine_id]
        # Define the failure point as the minimum RUL value in the dataset for the engine
        failure_point = engine_data['RUL'].min()
        # Filter the data within the sequence window of the failure point
        window_data = engine_data[engine_data['RUL'] <= failure_point + sequence_length]
        # Reset index to align data from the start
        window_data = window_data.reset_index(drop=True)
        return window_data[['cycle'] + cols]

    # Get data for all four engine IDs
    data1 = get_window_data_for_engine(engine_id1)
    data2 = get_window_data_for_engine(engine_id2)
    data3 = get_window_data_for_engine(engine_id3)
    data4 = get_window_data_for_engine(engine_id4)
    
    # Ensure all datasets start from index 0 for comparison
    max_len = max(len(data1), len(data2), len(data3), len(data4))
    data1 = data1.reindex(range(max_len)).reset_index(drop=True)
    data2 = data2.reindex(range(max_len)).reset_index(drop=True)
    data3 = data3.reindex(range(max_len)).reset_index(drop=True)
    data4 = data4.reindex(range(max_len)).reset_index(drop=True)
    
    # Plot sensor data for the specified columns
    num_cols = len(cols)
    plt.figure(figsize=(20, 5 * num_cols))
    
    for i, column in enumerate(cols):
        plt.subplot(num_cols, 1, i + 1)
        plt.plot(data1.index, data1[column], label=f'Engine ID {engine_id1}', color='blue', alpha=0.7)
        plt.plot(data2.index, data2[column], label=f'Engine ID {engine_id2}', color='red', alpha=0.7)
        plt.plot(data3.index, data3[column], label=f'Engine ID {engine_id3}', color='green', alpha=0.7)
        plt.plot(data4.index, data4[column], label=f'Engine ID {engine_id4}', color='orange', alpha=0.7)
        plt.title(f'Sensor {column}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
    plt.suptitle(f'Sensor Data Comparison for Engine IDs {engine_id1}, {engine_id2}, {engine_id3}, and {engine_id4}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
def group_and_aggregate(df, group_column, columns_to_use):
    """
    Groups the DataFrame by the specified column and computes various statistics for each group.
    
    Parameters:
    df (pd.DataFrame): The DataFrame.
    group_column (str): The column name to group by.
    columns_to_use (list): List of column names to compute statistics for.
    
    Returns:
    pd.DataFrame: A DataFrame with aggregated statistics for each group.
    """
    # Define aggregation functions
    def calculate_mode(series):
        modes = series.mode()
        return modes[0] if not modes.empty else np.nan
    
    def calculate_median(series):
        return series.median()
    
    def calculate_mean(series):
        return series.mean()
    
    def calculate_std(series):
        return series.std()

    def calculate_min(series):
        return series.min()

    def calculate_max(series):
        return series.max()
    
    # Prepare the aggregation dictionary
    aggregation_functions = {
        column: [calculate_mode, calculate_median, calculate_mean, calculate_std, calculate_min, calculate_max]
        for column in columns_to_use
    }
    
    # Group by the specified column
    grouped = df.groupby(group_column).agg(aggregation_functions)
    
    # Flatten MultiIndex columns
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    
    return grouped.reset_index()


def detect_and_quantify_outliers(df, columns_to_check):
    """
    Detects and quantifies outliers in the specified columns of the DataFrame using the IQR method.

    Parameters:
    df (pd.DataFrame): The DataFrame.
    columns_to_check (list): List of column names to check for outliers.

    Returns:
    pd.DataFrame: A DataFrame with the count of outliers for each column.
    """
    outlier_summary = []

    for column in columns_to_check:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier boundaries
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers
        num_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]
        outlier_summary.append({
            'Column': column,
            'Number of Outliers': num_outliers,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound
        })
    
    return pd.DataFrame(outlier_summary)



def plot_boxplots(df, columns_to_plot, n_cols_per_row=3):
    """
    Plots boxplots for the specified columns in the DataFrame to visualize potential outliers.
    The plots are arranged in a grid with a specified number of columns per row.

    Parameters:
    df (pd.DataFrame): The DataFrame.
    columns_to_plot (list): List of column names to plot.
    n_cols_per_row (int): Number of plots to display per row.
    """
    num_cols = len(columns_to_plot)
    num_rows = int(np.ceil(num_cols / n_cols_per_row))
    
    fig, axes = plt.subplots(num_rows, n_cols_per_row, figsize=(5 * n_cols_per_row, 5 * num_rows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration
    
    # Plot each column
    for i, column in enumerate(columns_to_plot):
        sns.boxplot(x=df[column], ax=axes[i])
        axes[i].set_title(f'Boxplot of {column}')
    
    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

def missing_values_report(df):
    """
    Returns a DataFrame indicating the number and percentage of missing values in each column.

    Parameters:
    df (pd.DataFrame): The DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with columns 'Column', 'Missing Values', and '% of Total Values'.
    """
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    missing_report = pd.DataFrame({'Column': df.columns,
                                   'Missing Values': missing_count,
                                   '% of Total Values': missing_percentage})
    return missing_report


def perform_pca_and_plot(df, columns_to_use):
    """
    Performs PCA on the specified columns of the DataFrame, plots the cumulative explained variance 
    as a bar graph with values on top of each bar, and plots the principal components in both 2D and 3D.

    Parameters:
    df (pd.DataFrame): The DataFrame.
    columns_to_use (list): List of column names to use for PCA.
    """
    # Select the specific columns
    df_subset = df[columns_to_use]

    # Standardize the data
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(df_subset)

    # Number of components to use is the minimum of the number of columns or the maximum PCA components requested
    n_components = min(len(columns_to_use), len(df_subset.columns))
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_standardized)
    
    # Plot cumulative explained variance as a bar graph
    def plot_cumulative_explained_variance(pca):
        cumulative_variance = pca.explained_variance_ratio_.cumsum()
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(1, len(cumulative_variance) + 1), cumulative_variance, alpha=0.7, color='skyblue')

        # Add text annotations
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', ha='center', va='bottom')

        plt.title('Cumulative Explained Variance by Principal Components')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.xticks(range(1, len(cumulative_variance) + 1))
        plt.ylim(0, 1)
        plt.show()

    plot_cumulative_explained_variance(pca)
    
    # Plot principal components (2D and 3D)
    def plot_principal_components(pc, n_components):
        if n_components >= 2:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.scatter(pc[:, 0], pc[:, 1], alpha=0.5)
            plt.title('Principal Components (2D)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')

        if n_components >= 3:
            plt.figure(figsize=(12, 10))
            ax = plt.axes(projection='3d')
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], alpha=0.5)
            ax.set_title('Principal Components (3D)')
            ax.set_xlabel('Principal Component 1')
            ax.set_ylabel('Principal Component 2')
            ax.set_zlabel('Principal Component 3')

        plt.show()

    plot_principal_components(principal_components, n_components)
def plot_correlation_matrix(df, threshold=0.8):
    """
    Plots the correlation matrix for numerical columns in the DataFrame and prints pairs of columns
    with correlation values above the specified threshold.

    Parameters:
    df (pd.DataFrame): The DataFrame.
    threshold (float): The correlation threshold for filtering significant values. Default is 0.8.
    """
    corr = df.corr()
    
    # Print pairs of columns with correlation values above the threshold
    print("Pairs of columns with correlation values above the threshold:")
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) >= threshold:
                print(f"{corr.columns[i]} vs {corr.columns[j]}: {corr.iloc[i, j]:.2f}")
    
    # Plot the correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
    plt.title('Correlation Matrix')
    plt.show()

def cuentaDistintos(datos):
    """
    Cuenta valores distintos en cada variable numerica de un DataFrame.

    Args:
        datos (DataFrame): El DataFrame que contiene los datos.

    Returns:
        Dataframe: Un DataFrame con las variables y valores distintos en cada una de ellas
    """
    # Seleccionar las columnas numéricas en el DataFrame
    numericas = datos.select_dtypes(include=['int', 'int32', 'int64','float', 'float32', 'float64'])
    
    # Calcular la cantidad de valores distintos en cada columna numérica
    resultados = numericas.apply(lambda x: len(x.unique()))
    
    # Crear un DataFrame con los resultados
    resultado = pd.DataFrame({'Columna': resultados.index, 'Distintos': resultados.values})
    
    return resultado
def unique_values_report(df, column_name):
    """
    Returns a string indicating the number of unique values, minimum value, and maximum value 
    in the specified column of the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame.
    column_name (str): The name of the column.

    Returns:
    str: A formatted string indicating the number of unique values, minimum value, and maximum value in the column.
    """
    unique_count = df[column_name].nunique()
    min_value = df[column_name].min()
    max_value = df[column_name].max()
    report = f"Column '{column_name}' has {unique_count} unique values. Min value: {min_value}, Max value: {max_value}"
    print(report)
    return 

# function to reshape features into (samples, time steps, features) 
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]

def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]


def plot_confusion_matrix(model, seq_array, y_true, verbose=1, batch_size=200):
    # Make predictions
    y_pred_prob = model.predict(seq_array, verbose=verbose, batch_size=batch_size)

    # Convert probabilities to class labels
    y_pred = (y_pred_prob > 0.5).astype(int)  # Assuming binary classification with sigmoid activation

    # Convert y_true to a numpy array if it isn't already
    y_true = np.array(y_true).flatten()  # Flatten in case y_true is multi-dimensional

    # Flatten y_pred to match the shape of y_true
    y_pred = y_pred.flatten()

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Display metrics in a tabular format
    metrics_table = [
        ["Accuracy", accuracy],
        ["Precision", precision],
        ["Recall", recall],
        ["F1 Score", f1]
    ]
    
    print("\nMetrics:")
    print(tabulate(metrics_table, headers=["Metric", "Score"], tablefmt="grid"))

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'], 
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def build_lstm_network(sequence_length, nb_features, nb_out, lstm_units_1, lstm_units_2, dropout_rate, optimizer_name):
    model = Sequential()

    # Add the first LSTM layer
    model.add(LSTM(
        units=lstm_units_1,
        input_shape=(sequence_length, nb_features),
        return_sequences=True))
    model.add(Dropout(dropout_rate))

    # Add the second LSTM layer
    model.add(LSTM(
        units=lstm_units_2,
        return_sequences=False))
    model.add(Dropout(dropout_rate))

    # Add the Dense output layer
    model.add(Dense(units=nb_out, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=optimizer_name, metrics=['accuracy'])

    
    return model

def create_model(units_lstm1, units_lstm2, dropout_rate):
    sequence_cols = [
    'setting1', 'setting2', 'setting3', 'cycle_norm',
    's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9',
    's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
    's18', 's19', 's20', 's21'
    ]
    sequence_length = 50
    model = Sequential()
    model.add(LSTM(units=units_lstm1, return_sequences=True, input_shape=(sequence_length, len(sequence_cols))))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units_lstm2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # Output layer for regression

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_and_plot(model, X, y, epochs=20, batch_size=64, validation_split=0.1):
    # Fit the model with provided parameters
    history = model.fit(
        X, 
        y, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=validation_split
    )
    
    # Evaluate the model
    loss = model.evaluate(X, y, verbose=0)
    print(f'Loss: {loss}')

    # Calculate predictions and MSE
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f'Mean Squared Error: {mse}')

    # Create a DataFrame to compare real vs. predicted RUL
    results_df = pd.DataFrame({
        'True RUL': y.flatten(),
        'Predicted RUL': predictions.flatten()
    })

    # Print metrics table
    print("\nMetrics:")
    print(f"{'Loss':<20} {loss:.4f}")
    print(f"{'Mean Squared Error':<20} {mse:.4f}")

    # Plot training & validation loss values
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot Mean Squared Error
    plt.figure(figsize=(12, 6))
    plt.scatter(results_df['True RUL'], results_df['Predicted RUL'], alpha=0.5)
    plt.title('True vs. Predicted RUL')
    plt.xlabel('True RUL')
    plt.ylabel('Predicted RUL')
    plt.plot([results_df['True RUL'].min(), results_df['True RUL'].max()],
             [results_df['True RUL'].min(), results_df['True RUL'].max()],
             color='red', linestyle='--')
    plt.show()

    return results_df

def create_sequences_lstm_rul(df, sequence_length, feature_cols, label_col):
    sequences = []
    labels = []
    
    # Group by 'id' to handle sequences for each machine
    for machine_id in df['id'].unique():
        machine_df = df[df['id'] == machine_id]
        
        # Ensure data is sorted by 'cycle'
        machine_df = machine_df.sort_values('cycle')
        
        # Create sequences and corresponding labels
        for start in range(len(machine_df) - sequence_length):
            end = start + sequence_length
            sequence = machine_df[feature_cols].iloc[start:end].values
            label = machine_df[label_col].iloc[end]  # RUL at the end of the sequence
            
            sequences.append(sequence)
            labels.append(label)
    
    return np.array(sequences), np.array(labels)

def evaluate_rul_performance_lstm(y_true, y_pred):
    """
    Evaluate the performance of RUL predictions by plotting:
    1. A scatter plot of True RUL vs. Predicted RUL.
    2. A line plot of True RUL and Predicted RUL over time/sample index.

    Parameters:
    - y_true: array-like, true RUL values
    - y_pred: array-like, predicted RUL values

    Returns:
    - None
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Scatter plot of True RUL vs. Predicted RUL
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel('True RUL')
    plt.ylabel('Predicted RUL')
    plt.title('True RUL vs. Predicted RUL')
    plt.grid(True)
    
    # Line plot of True RUL and Predicted RUL
    plt.subplot(1, 2, 2)
    plt.plot(y_true, label='True RUL', linestyle='-', marker='o', markersize=3, color='blue')
    plt.plot(y_pred, label='Predicted RUL', linestyle='-', marker='x', markersize=3, color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('RUL')
    plt.title('True RUL and Predicted RUL Over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

