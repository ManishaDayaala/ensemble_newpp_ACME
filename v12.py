#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import shutil
import pandas as pd
from datetime import datetime
import streamlit as st
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib


# Set a random seed for reproducibility
def set_random_seed(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)

# Define the main folder path
MAINFOLDER = r"./APPdata"

# Create other paths relative to the main folder
training_file_path = os.path.join(MAINFOLDER, "Training", "Training.xlsx")  # FIXED TRAINING DATA
test_file_path = os.path.join(MAINFOLDER, "24hrData", "Dailydata.xlsx")  # DAILY DATA
excel_file_path = os.path.join(MAINFOLDER, "Breakdownrecords.xlsx")  # Recording excel for BD
folderpath = os.path.join(MAINFOLDER, "TemporaryData")  # Temporary dump files collector



# Define the path to save models within the main folder
model_folder_path = os.path.join(MAINFOLDER, "Models")


uploaded_files = []  # List to keep track of uploaded files

# Streamlit UI
st.title("Breakdown Predictor")
st.markdown("Upload your files, and they will be preprocessed accordingly.")


# Initialize file uploader key in session state
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

# File uploader
uploaded_files = st.file_uploader("Upload your files", accept_multiple_files=True, key=str(st.session_state["file_uploader_key"]))


# Show status
status_placeholder = st.empty()

# Function to clear old files from the folder
def clear_saved_files():
    try:
        # Clear old files in the folder
        for filename in os.listdir(folderpath):
            file_path = os.path.join(folderpath, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Remove the file
            except Exception as e:
                status_placeholder.error(f"Error clearing old files: {e}")
                return
        status_placeholder.success("Saved files cleared successfully!")
    except Exception as e:
        status_placeholder.error(f"Error: {e}")

# Function to handle file saving (clear old files before saving new ones)
def save_files(uploaded_files):
    try:
        if not uploaded_files:
            status_placeholder.error("No files to save!")
            return

        # Clear old files in the folder before saving new files
        clear_saved_files()

        # Save each file from the uploaded list to the target folder
        for file in uploaded_files:
            with open(os.path.join(folderpath, file.name), "wb") as f:
                f.write(file.getbuffer())

        status_placeholder.success("Files saved successfully!")
        # Clear uploaded files from the interface after saving   addedd extra
        st.session_state["file_uploader_key"] += 1


    except Exception as e:
        status_placeholder.error(f"Error: {e}")



# Clear previous uploaded files display automatically before handling new uploads
if st.button("Save Files"):
    if uploaded_files:
        st.session_state['uploaded_files'] = None  # Reset session state to clear display
        st.session_state['uploaded_files'] = uploaded_files  # Store new uploads in session state
        save_files(st.session_state['uploaded_files'])  # Clear old files and save new ones
    else:
        st.error("Please upload files first.")



#if st.button("Save Files"):
#    if uploaded_files:
#        save_files(uploaded_files)
#    else:
#        st.error("Please upload files first.")







################# data preprocessing         ###################


import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

def process_data():
    
    # Define the input file (only one file in the folder)
    input_file_name = os.listdir(folderpath)[0]  # Assuming only one file in the folder
    input_file_path = os.path.join(folderpath, input_file_name)

    # Check if the input file exists
    if not os.path.isfile(input_file_path):
        st.error(f"Input file '{input_file_name}' does not exist!")
        return

    # List of 12 unique asset names
    assets_list = [
        "A1 GM 1 GB IP DE", "A1 GM1 MDE", "A1 GM 2 GB IP DE", "A1 GM2 MDE",
        "A1 GM 3 GB IP DE", "A1 GM3 MDE", "A1 GM 4 GB IP DE", "A1 GM4 MDE",
        "A1 GM 5 GB IP DE", "A1 GM5 MDE", "A1 GM 6 GB IP DE", "A1 GM6 MDE"
    ]

    # Columns to extract for each asset, corresponding to F, I, L, O, R, U
    required_column_indices = [5, 8, 11, 14, 17, 20]  # 0-based indices for F, I, L, O, R, U
    required_column_names = ['a2', 'vv2', 'av2', 'hv2', 't2', 'd2']

    # Check if the output folder exists, if not, create it
    #if not os.path.isdir(test_file_path):
        #os.makedirs(test_file_path)
        #st.info(f"Output folder '{test_file_path}' created!")

    # Load the input file
    input_df = pd.read_excel(input_file_path)

    # Initialize an empty DataFrame to store combined data
    output_df = pd.DataFrame()

    # Loop over each asset in assets_list
    for asset_name in assets_list:
        # Find rows where Column B (index 1) matches the asset_name
        asset_rows = input_df[input_df.iloc[:, 1] == asset_name]
        
        # Check if any rows were found
        if not asset_rows.empty:
            # Parse the date and time from Column C (index 2)
            asset_rows['DateTime'] = pd.to_datetime(asset_rows.iloc[:, 2], format='%d-%m-%Y %H:%M')
            
            # Identify the earliest start time in the data for this asset
            start_time = asset_rows['DateTime'].min().replace(hour=5, minute=30)
            end_time = start_time + timedelta(days=1, hours=0, minutes=0)
            
            # Filter rows within this 24-hour window (from earliest 5:30 AM to the next day 5:30 AM)
            filtered_rows = asset_rows[(asset_rows['DateTime'] >= start_time) & (asset_rows['DateTime'] <= end_time)]
            
            # Select only the first 49 rows if there are more than 49 available
            filtered_rows = filtered_rows.head(49)
            
            # Collect only the specified columns (F, I, L, O, R, U) for the 49 rows
            data_for_asset = filtered_rows.iloc[:, required_column_indices].values
            data_for_asset = pd.DataFrame(data_for_asset, columns=required_column_names)
            
            # Fill any missing rows with 0s if there are fewer than 49 rows
            if len(data_for_asset) < 49:
                missing_rows = 49 - len(data_for_asset)
                data_for_asset = pd.concat([data_for_asset, pd.DataFrame(0, index=range(missing_rows), columns=required_column_names)], ignore_index=True)
        else:
            # If no rows found for this asset, fill with 0s for all columns
            data_for_asset = pd.DataFrame(0, index=range(49), columns=required_column_names)

        # Rename columns to reflect asset-specific names (e.g., "a2" becomes "A1 GM 1 GB IP DE_a2")
        data_for_asset.columns = [f"{asset_name}_{col}" for col in required_column_names]

        # Concatenate the data for this asset horizontally to the output DataFrame
        output_df = pd.concat([output_df, data_for_asset], axis=1)

    # Generate Date, Time, and Sr No columns at 30-minute intervals
    date_list = [(start_time + timedelta(minutes=30 * i)).strftime('%d-%m-%Y') for i in range(49)]
    time_list = [(start_time + timedelta(minutes=30 * i)).strftime('%I:%M %p') for i in range(49)]
    sr_no_list = list(range(1, 50))


    # Insert Date, Time, and Sr No columns into the final output DataFrame
    output_df.insert(0, 'Date', date_list)
    output_df.insert(1, 'Time', time_list)
    output_df.insert(2, 'Sr No', sr_no_list)

    # Add an empty 'Code' column at the end
    output_df['Code'] = ''


    # Save the processed data using ExcelWriter
    with pd.ExcelWriter(test_file_path, engine='openpyxl') as writer:
        output_df.to_excel(writer, index=False)

    
    # Display success message when all files are processed
    st.info(f"Data has been processed and saved")


# Create a button to trigger the process
if st.button('Preprocess Data'):
    process_data()


##########################  Classification ###############################

import tensorflow as tf
import random
import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import VotingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Set random seed for reproducibility
def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

# Define the training function
def train_ensemble_model(training_file_path, model_folder_path):
    def load_data(file_path):
        df = pd.read_excel(file_path)
        X = df.iloc[:, 3:-1].values  # Features (assuming columns 3 to second last)
        y = df['Code'].values  # Target column
        return X, y

    def preprocess_data(X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Save the scaler
        joblib.dump(scaler, os.path.join(model_folder_path, "scaler_new.pkl"))

        # Split into training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.01, random_state=42, shuffle=True)

        # Handle imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        return X_resampled, X_test, y_resampled, y_test

    def build_nn_model():
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(X_resampled.shape[1],)))
        model.add(Dropout(0.2))
        #model.add(Dense(64, activation='relu'))
        #model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(4, activation='softmax'))  # 4 output units for the 4 classes
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Use sparse_categorical_crossentropy
        return model

    # Set random seed
    set_random_seed()

    # Load and preprocess data
    X, y = load_data(training_file_path)
    X_resampled, X_test, y_resampled, y_test = preprocess_data(X, y)

    # Class weights for Keras model
    class_weights_nn = {0: 1.0, 1: 30, 2: 30, 3: 30}

    # Build Keras model
    nn_model = KerasClassifier(model=build_nn_model, epochs=50, batch_size=32, verbose=0, class_weight=class_weights_nn)

    # Calculate sample weights for XGBoost
    sample_weights = np.array([class_weights_nn[int(label)] for label in y_resampled])

    # XGBoost model
    xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4, eval_metric='mlogloss', sample_weight=sample_weights, random_state=42)

    # Ensemble model
    ensemble_model = VotingClassifier(estimators=[
        ('xgb', xgb_model),
        ('nn', nn_model)
    ], voting='soft')

    # Train the ensemble model
    ensemble_model.fit(X_resampled, y_resampled)

    # Save the trained model
    joblib.dump(ensemble_model, os.path.join(model_folder_path, "ensemble_model_new.pkl"))
    st.success("Ensemble model training completed and saved!")

# Define the prediction function
def predict_ensemble(test_file_path, model_folder_path):
    def load_test_data(file_path):
        df = pd.read_excel(file_path)
        X_test = df.iloc[:, 3:-1].values
        return df, X_test

    def preprocess_test_data(X_test):
        scaler = joblib.load(os.path.join(model_folder_path, "scaler_new.pkl"))
        X_test_scaled = scaler.transform(X_test)
        return X_test_scaled

    def predict(X_test_scaled):
        ensemble_model = joblib.load(os.path.join(model_folder_path, "ensemble_model_new.pkl"))
        predictions = ensemble_model.predict(X_test_scaled)
        return predictions

    set_random_seed()

    try:
        df, X_test = load_test_data(test_file_path)
        X_test_scaled = preprocess_test_data(X_test)
        predictions = predict(X_test_scaled)

        breakdown_codes = ["Code 0", "Code 1", "Code 2", "Code 3"]
        predicted_labels = [breakdown_codes[p] for p in predictions]

        # Check if any non-zero breakdown code (Code 1, 2, or 3) is predicted
        non_zero_codes = [code for code in predicted_labels if "Code 1" in code or "Code 2" in code or "Code 3" in code]
        
        if non_zero_codes:
            unique_non_zero_codes = set(non_zero_codes)
            return f"Breakdown of {', '.join(unique_non_zero_codes)} might occur."
        else:
            return "No BD predicted"
    except Exception as e:
        return f"Error: {e}"

# Streamlit app UI
st.title("Breakdown Code Classification")

if st.button("Check BD Classification"):
    #training_file_path = "path/to/training_file.xlsx"  # Update the path
    #test_file_path = "path/to/test_file.xlsx"  # Update the path
    #model_folder_path = "path/to/model_folder"  # Update the path

    with st.spinner("Training the model and making predictions..."):
        #train_ensemble_model(training_file_path, model_folder_path)  # Train the model
        result = predict_ensemble(test_file_path, model_folder_path)  # Predict breakdown

    st.write(result)
    st.success("Prediction complete!")


###########################                                    #######################################

# Use an expander to provide breakdown code information
with st.expander("Breakdown Classification and Codes", expanded=True):
    st.markdown("""
    Each breakdown type is assigned a unique code to simplify identification. Here’s what each code represents:

    - **Code 1: Contact Wheel and Spindle Issues**  
      Issues specifically related to the Contact Wheel or Spindle mechanisms of the machine.
    
    - **Code 2: Idler Wheel, Belt Tension Cylinder, and Work Rest Issues**  
      Covers problems with the Idler Wheel, Belt Tension Cylinder, and Work Rest components.
    
    - **Code 3: Regulating Wheel, Ball Screw Rod, and LM Rail Issues**  
      Applies to issues with the Regulating Wheel, Ball Screw Rod, or LM Rail systems.
    """)

       



################################        time prediction             #############################

import streamlit as st
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from datetime import datetime
import numpy as np


# Function to set random seed for reproducibility
def set_random_seed(seed=42):
    np.random.seed(seed)

# Define the training function
def train_model(training_file_path):
    def load_data(file_path):
        df = pd.read_excel(file_path, sheet_name="Time")
        X = df.iloc[:, 1:72].values
        y = df.iloc[:, 73].values
        return X, y

    def preprocess_data(X, y):
        mask = y < 90  # Time to breakdown less than 72 hours
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        # Use a fixed random_state to ensure reproducibility
        X_train, X_val, y_train, y_val = train_test_split(X_filtered, y_filtered, test_size=0.01, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        joblib.dump(scaler, os.path.join(model_folder_path, 'scalerfinT.pkl'))
        return X_train_scaled, X_val_scaled, y_train, y_val

    def build_model(input_shape):
        model = Sequential()
        model.add(Dense(128, input_dim=input_shape, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    # Set random seed for reproducibility
    set_random_seed()

    X, y = load_data(training_file_path)
    X_train, X_val, y_train, y_val = preprocess_data(X, y)
    model = build_model(X_train.shape[1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping])
    model.save(os.path.join(model_folder_path, 'trained_modelFINT.h5'))

# Define the prediction function
def predict_time(test_file_path):
    def load_test_data(file_path):
        df = pd.read_excel(file_path)
        serial_numbers = df.iloc[:, 2].values
        times = df.iloc[:, 1].values
        X_test = df.iloc[:, 3:74].values
        return df, X_test, serial_numbers, times

    def preprocess_test_data(X_test):
        scaler = joblib.load(os.path.join(model_folder_path, 'scalerfinT.pkl'))
        X_test_scaled = scaler.transform(X_test)
        return X_test_scaled

    def predict_time_to_breakdown(X_test_scaled):
        model = load_model(os.path.join(model_folder_path, 'trained_modelFINT.h5'))
        predictions = model.predict(X_test_scaled)
        return predictions

    def calculate_time_difference(times, predictions):
        time_to_breakdown_with_time = []
        base_time = datetime.strptime(times[0],'%I:%M %p')
        for time_str, prediction in zip(times, predictions):
            time_obj = datetime.strptime(time_str, '%I:%M %p')
            #midnight = datetime.combine(time_obj.date(), datetime.min.time())
            time_difference = (time_obj - base_time).total_seconds() / 3600
            adjusted_time_to_bd = prediction[0] + time_difference
            time_to_breakdown_with_time.append(adjusted_time_to_bd)
        return time_to_breakdown_with_time

   
    def find_minimum_and_maximum_time(time_to_breakdown_with_time):
        # Filter out negative times
        positive_times = [time for time in time_to_breakdown_with_time if time >= 0]
    
        #if not positive_times:
            #return "No positive breakdown times available."
    
        min_time = min(positive_times)
        max_time = max(time_to_breakdown_with_time)
        
        return min_time, max_time
    
   
    try:
        # Load and preprocess the test data
        df, X_test, serial_numbers, times = load_test_data(test_file_path)
        X_test_scaled = preprocess_test_data(X_test)

        # Make predictions
        predictions = predict_time_to_breakdown(X_test_scaled)
        predictions_with_time = calculate_time_difference(times, predictions)

        # Find the minimum and maximum predicted times
        min_time, max_time = find_minimum_and_maximum_time(predictions_with_time)

        # Format the output as a range in hours
        return f"Breakdown time range (w.r.t 5:30 AM ): {min_time:.2f} to {max_time:.2f} hours"
    except Exception as e:
        return f"Error: {e}"

    


# Streamlit app UI
st.title("Time Prediction")

# Button to train the model and predict time
if st.button("Predict Time"):
    # Train the model (if needed) and predict time
    with st.spinner("Training the model and making predictions..."):
        #train_model(training_file_path)  # Train the model (use predefined training data)
        result = predict_time(test_file_path)  # Predict time using predefined test data
    
    st.write(f"Predicted Time to Breakdown: {result}")
    st.success("Prediction complete!")





#################### Classification    ###############################

###import streamlit as st
###import os
###import pandas as pd
###import numpy as np
###from tensorflow.keras.models import Sequential, load_model
###from tensorflow.keras.layers import Dense, Dropout
###from tensorflow.keras.callbacks import EarlyStopping
###from sklearn.model_selection import train_test_split
###from sklearn.preprocessing import StandardScaler
###import joblib
###
###
#### Function to set random seed for reproducibility
###def set_random_seed(seed=42):
###    np.random.seed(seed=42)
###
#### Define the training function
###def train_model_classification(training_file_path):
###    def load_data(file_path):
###        df = pd.read_excel(file_path, sheet_name="Classification")
###        X = df.iloc[:, 3:-1].values  # Assuming features are from column 1 to second last
###        y = df.iloc[:, -1].values  # Target is in the last column
###        return X, y
###
###    def preprocess_data(X, y):
###        # Scale the input features
###        scaler = StandardScaler()
###        X_scaled = scaler.fit_transform(X)
###        
###        # Save the scaler for future use
###        #joblib.dump(scaler, scaler_path)
###        joblib.dump(scaler, os.path.join(model_folder_path, "scaler_classification.pkl"))
###        
###        # Split data into training and validation sets
###        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.01, random_state=42)
###        return X_train, X_val, y_train, y_val
###
###    def build_model(input_shape):
###        # Build the neural network model
###        model = Sequential()
###        model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
###        model.add(Dropout(0.2))
###        model.add(Dense(64, activation='relu'))
###        model.add(Dropout(0.2))
###        model.add(Dense(32, activation='relu'))
###        model.add(Dense(4, activation='softmax'))  # 4 output units for the 4 classes
###        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
###        return model
###
###    # Set random seed for reproducibility
###    set_random_seed()
###
###    # Load and preprocess the data
###    X, y = load_data(training_file_path)
###    X_train, X_val, y_train, y_val = preprocess_data(X, y)
###    
###    class_weight_nn = {0: 1.0, 1: 240, 2: 260, 3: 280}
###
###    # Build the model
###    model = build_model(X_train)
###
###    
###
###    # Train the model
###    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100,     batch_size=32,class_weight=class_weight_nn)
###
###    # Save the trained model
###    #model.save(model_path)
###    # model = joblib.load(os.path.join(model_folder_path, 'ensemble_modelFINP.pkl'))
###    model.save(os.path.join(model_folder_path, 'ensemble_modelFINP.h5'))
###
###    # joblib.dump(model, os.path.join(model_folder_path, 'ensemble_modelFINP.pkl'))
###    st.success("Model training completed and saved!")
###
#### Define the prediction function
###def predict_breakdown(test_file_path):
###    def load_test_data(file_path):
###        df = pd.read_excel(file_path)
###        X_test = df.iloc[:, 3:-1].values  # Features from column 1 to second last
###        return df, X_test
###
###    def preprocess_test_data(X_test):
###        # Load the scaler and transform the test data
###        #scaler = joblib.load(scaler_path)
###        scaler = joblib.load(os.path.join(model_folder_path, "scaler_classification.pkl"))
###        X_test_scaled = scaler.transform(X_test)
###        return X_test_scaled
###
###    def predict_classification(X_test_scaled):
###        # Load the trained model and make predictions
###        #model = load_model(model_path)
###        model = load_model(os.path.join(model_folder_path, 'ensemble_modelFINP.h5'))
###       # model = joblib.load(model, os.path.join(model_folder_path, 'ensemble_modelFINP.pkl'))
###        predictions = model.predict(X_test_scaled)
###        return predictions
###
###    # Set random seed for reproducibility
###    set_random_seed()
###
###    try:
###        # Load and preprocess the test data
###        df, X_test = load_test_data(test_file_path)
###        X_test_scaled = preprocess_test_data(X_test)
###
###        # Make predictions
###        predictions = predict_classification(X_test_scaled)
###
###        # Get the predicted class labels (highest probability class)
###        predicted_classes = np.argmax(predictions, axis=1)
###
###        # Map the predicted classes to breakdown codes (0, 1, 2, 3)
###        breakdown_codes = ["Code 0", "Code 1", "Code 2", "Code 3"]
###        predicted_labels = [breakdown_codes[i] for i in predicted_classes]
###
###        # Check if any non-zero breakdown code (Code 1, 2, or 3) is predicted
###        non_zero_codes = [code for code in predicted_labels if "Code 1" in code or "Code 2" in code or "Code 3" in code]
###
###        
###        # Only return results if non-zero codes are predicted
###        if non_zero_codes:
###            unique_non_zero_codes = set(non_zero_codes)
###            num_unique_non_zero_codes = len(unique_non_zero_codes)
###            df['Predicted Breakdown'] = predicted_labels
###            return (
###                #df[['Predicted Breakdown']], Count: {num_unique_non_zero_codes},
###                f"Breakdown of {', '.join(unique_non_zero_codes)} might occur."
###            )
###        else:
###            # Return a message indicating no breakdown was predicted
###            return None, "No BD predicted"
###    except Exception as e:
###        return f"Error: {e}", None
###
#### Streamlit app UI
###st.title("Breakdown Code Classification")
###
###
###if st.button("check BD classification"):
###    # Train the model (if needed) and predict time
###    with st.spinner("Training the model and making predictions..."):
###        #train_model_classification(training_file_path)  # Train the model (use predefined training data)
###        result = predict_breakdown(test_file_path)  # Predict time using predefined test data
###    
###    st.write(f"classified breakdown: {result}")
###    st.success("Prediction complete!")




################breakdown records###########################

import streamlit as st
import pandas as pd
from datetime import datetime

# Path to the Excel file
#excel_file_path = "breakdown_data.xlsx"

# Function to save breakdown data to Excel
def save_breakdown_data():
    date = st.session_state.date_entry.strftime("%d-%m-%y")
    time = f"{st.session_state.hour_combobox}:{st.session_state.minute_combobox} {st.session_state.am_pm_combobox}"
    code = st.session_state.code_entry
    
    if not code:
        st.session_state.status = "Please fill the Breakdown Code!"
        st.session_state.status_color = "red"
        return

    try:
        df = pd.read_excel(excel_file_path)
        new_row = pd.DataFrame([[date, time, code]], columns=["Date", "Time", "Code"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(excel_file_path, index=False)
        st.session_state.status = "Breakdown data saved successfully!"
        st.session_state.status_color = "green"
    except Exception as e:
        st.session_state.status = f"Error: {e}"
        st.session_state.status_color = "red"

# Function to clear the breakdown input fields
def clear_breakdown_fields():
    st.session_state.date_entry = datetime.now()
    st.session_state.hour_combobox = '12'
    st.session_state.minute_combobox = '00'
    st.session_state.am_pm_combobox = 'AM'
    st.session_state.code_entry = ''
    st.session_state.status = "Fields cleared!"
    st.session_state.status_color = "blue"

# Streamlit UI Setup
def display_ui():
    # Initialize session state if not already initialized
    if 'status' not in st.session_state:
        st.session_state.status = ""
        st.session_state.status_color = "black"
        st.session_state.date_entry = datetime.now()
        st.session_state.hour_combobox = '12'
        st.session_state.minute_combobox = '00'
        st.session_state.am_pm_combobox = 'AM'
        st.session_state.code_entry = ''

    st.title("Breakdown Record")

    # Date input
    st.session_state.date_entry = st.date_input("Date", value=st.session_state.date_entry)
    
    # Time selection
    time_column1, time_column2, time_column3 = st.columns(3)
    with time_column1:
        st.session_state.hour_combobox = st.selectbox("Hour", options=[f"{i:02d}" for i in range(1, 13)], index=int(st.session_state.hour_combobox)-1)
    with time_column2:
        st.session_state.minute_combobox = st.selectbox("Minute", options=[f"{i:02d}" for i in range(0, 60, 5)], index=int(st.session_state.minute_combobox)//5)
    with time_column3:
        st.session_state.am_pm_combobox = st.selectbox("AM/PM", options=["AM", "PM"], index=["AM", "PM"].index(st.session_state.am_pm_combobox))

    # Breakdown code input
    st.session_state.code_entry = st.text_input("Breakdown Code", value=st.session_state.code_entry)

    # Status display (Feedback to user)
    st.markdown(f"<p style='color:{st.session_state.status_color};'>{st.session_state.status}</p>", unsafe_allow_html=True)

    # Buttons for saving and clearing
    col1, col2 = st.columns(2)
    with col1:
        save_button = st.button("Save Breakdown")
        if save_button:
            save_breakdown_data()
    with col2:
        clear_button = st.button("Clear Fields")
        if clear_button:
            clear_breakdown_fields()

# Run the UI display
if __name__ == "__main__":
    display_ui()





