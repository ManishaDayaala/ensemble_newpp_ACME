
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
threshold_file_path = os.path.join(MAINFOLDER,"Thresholds.xlsx") #

# Define the path to save models within the main folder
model_folder_path = os.path.join(MAINFOLDER, "Models")

#....CHANGED...........................................................................................................
if "check_bd_clicked" not in st.session_state:
    st.session_state["check_bd_clicked"] = False
if "bd_output" not in st.session_state:
    st.session_state["bd_output"] = ""

uploaded_files = []  # List to keep track of uploaded files

# Streamlit UI
st.title("Breakdown Predictor")
st.markdown("Upload your files, and they will be preprocessed accordingly.")

# File Upload Section
uploaded_files = st.file_uploader("Upload Excel files", type=['xlsx'], accept_multiple_files=True)

#uploaded_files = st.file_uploader("Drop files here:", accept_multiple_files=True, on_change=lambda: on_file_drop(uploaded_files))


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



# Initialize file uploader key in session state
if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

# File uploader
uploaded_files = st.file_uploader("Upload your files", accept_multiple_files=True, key=str(st.session_state["file_uploader_key"]))

# Clear previous uploaded files display automatically before handling new uploads
if st.button("Save Files"):
    if uploaded_files:
        st.session_state['uploaded_files'] = None  # Reset session state to clear display
        st.session_state['uploaded_files'] = uploaded_files  # Store new uploads in session state
        save_files(st.session_state['uploaded_files'])  # Clear old files and save new ones
    else:
        st.error("Please upload files first.")





###################### DATA PREPROCESSING   ############################


# Process files and apply preprocessing logic
def preprocess_files():
    try:
        # Step 1: Get all Excel files in the folder
        excel_files = [f for f in os.listdir(folderpath) if f.endswith('.xlsx')]

        # Step 2: Loop through each Excel file and preprocess
        for file in excel_files:
            file_path = os.path.join(folderpath, file)
            df = pd.read_excel(file_path)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns

            # Check for duplicate column names and give a warning if found
            duplicate_columns = df.columns[df.columns.duplicated()].tolist()
            if duplicate_columns:
                st.warning(f"File: {file}\nDuplicate columns found: {', '.join(duplicate_columns)}")

            # Step 4: Remove specific unnecessary columns
            columns_to_remove = [
                'plant_name', 'area_name', 'equipment_name', 'measurement_location_name',
                'avg_Vertical_velocity', 'avg_Axial_velocity', 'avg_Horizontal_velocity',
                'avg_total_acceleration', 'avg_audio', 'avg_temperature',
                'min_total_acceleration', 'min_Vertical_velocity', 'min_Axial_velocity',
                'min_Horizontal_velocity', 'min_temperature', 'min_audio'
            ]
            df.drop(columns=columns_to_remove, inplace=True, errors='ignore')

            # Step 5: Rename columns
            column_rename_map = {
                'max_total_acceleration': 'tot_acc',
                'max_Vertical_velocity': 'ver_vel',
                'max_Axial_velocity': 'ax_vel',
                'max_Horizontal_velocity': 'hor_vel',
                'max_temperature': 'temp',
                'max_audio': 'aud'
            }
            df.rename(columns=column_rename_map, inplace=True)

            # Step 6: Handle missing values for 'asset_name'
            if 'asset_name' in df.columns:
                df['asset_name'].fillna(method='ffill', inplace=True)
                df['asset_name'].fillna(method='bfill', inplace=True)

            # Step 7: Convert 'time' to datetime and remove timezone
            df['time'] = pd.to_datetime(df['time'], errors='coerce').dt.tz_localize(None)
            date_for_file = df['time'].dt.date.iloc[0]

            # Drop duplicates based on 'time' and keep the first occurrence
            df.drop_duplicates(subset='time', inplace=True)

            # Check for duplicate index values and show a warning
            if df['time'].duplicated().any():
                duplicate_rows = df[df['time'].duplicated()]
                warning_message = (
                    f"File: {file}\n"
                    f"Duplicate 'time' values found in rows:\n"
                    f"{duplicate_rows[['time']].to_string(index=False)}"
                )
                st.warning(warning_message)

            # Create a minute-wise time range
            start_time = pd.Timestamp(f"{date_for_file} 00:00:00")
            end_time = pd.Timestamp(f"{date_for_file} 23:59:00")
            full_time_range = pd.date_range(start=start_time, end=end_time, freq='T')

            # Set 'time' as the index and reindex with the full time range
            df.set_index('time', inplace=True)
            df = df.reindex(full_time_range)
            df.index.name = 'time'

            # Fill missing values
            non_numeric_cols = df.select_dtypes(exclude=['number']).ffill().bfill()
            numeric_cols = df.select_dtypes(include=['number']).fillna(0)
            resampled_numeric = numeric_cols.resample('10T', label='left', closed='left').max()
            resampled_non_numeric = non_numeric_cols.resample('10T').ffill()
            resampled_df = pd.concat([resampled_numeric, resampled_non_numeric], axis=1)
            resampled_df.reset_index(inplace=True)

            # Format 'time' to 'Date' and 'Time'
            resampled_df['Date'] = resampled_df['time'].dt.strftime('%d %b %Y')
            resampled_df['Time'] = resampled_df['time'].dt.strftime('%I:%M %p')
            resampled_df.insert(0, 'Sr No', range(1, len(resampled_df) + 1))
            ordered_columns = ['Date', 'Time', 'Sr No', 'tot_acc', 'ver_vel', 'ax_vel', 'hor_vel', 'temp', 'aud', 'asset_name']
            resampled_df = resampled_df[[col for col in ordered_columns if col in resampled_df.columns]]

            # Save the processed data back to the same file
            with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
                resampled_df.to_excel(writer, index=False)

        # Combining files
        asset_order = [
            "Grinding Machine 1 Gearbox", "Grinding Machine-1 Motor", "Grinding Machine 2 Gearbox",
            "Grinding Machine-2 Motor", "Grinding Machine 3 Gearbox", "Grinding Machine-3 Motor",
            "Grinding Machine 4 Gearbox", "Grinding Machine-4 Motor", "Grinding Machine 5 Gearbox",
            "Grinding Machine-5 Motor", "Grinding Machine 6 Gearbox", "Grinding Machine-6 Motor"
        ]

##combined_df = pd.DataFrame()
##for asset in asset_order:
##    for file in excel_files:
##        file_path = os.path.join(folderpath, file)
##        try:
##            df = pd.read_excel(file_path)
##            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
##        except Exception as e:
##            continu##
##        asset_name_values = df['asset_name'].iloc[1:3].values
##        if any(asset.strip() == value.strip() for value in asset_name_values):
##            df = df.drop(columns=['asset_name'], errors='ignore')
##            if combined_df.empty:
##                common_cols = ['Date', 'Time', 'Sr No']
##                combined_df = df[common_cols].copy(##
##            df = df.drop(columns=['Date', 'Time', 'Sr No'], errors='ignore')
##            combined_df = pd.concat([combined_df, df], axis=1##
##if not combined_df.empty:
##    combined_df.fillna(0, inplace=True)
##    combined_df['Code'] = ''
##    with pd.ExcelWriter(test_file_path, engine='openpyxl') as writer:
##        combined_df.to_excel(writer, index=False)
##    st.success("All files processed and combined successfully!")
##else:
##    st.error("No data found or processed. Please check files and asset names.")

        # Initialize an empty combined_df with common columns and asset-specific columns
        combined_columns = ['Date', 'Time', 'Sr No'] + ['tot_acc', 'ver_vel', 'ax_vel', 'hor_vel', 'temp', 'aud'] * len(asset_order)
        combined_df = pd.DataFrame(columns=combined_columns)

        excel_files = [f for f in os.listdir(folderpath) if f.endswith('.xlsx')]

        for asset in asset_order:
            asset_found = False

            for file in excel_files:
                file_path = os.path.join(folderpath, file)
                try:
                    df = pd.read_excel(file_path)
                    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                except Exception as e:
                    continue

                asset_name_values = df['asset_name'].iloc[1:3].values
                if any(asset.strip() == value.strip() for value in asset_name_values):
                    asset_found = True
                    df = df.drop(columns=['asset_name'], errors='ignore')
                    if combined_df.empty:
                        common_cols = ['Date', 'Time', 'Sr No']
                        combined_df = df[common_cols].copy()

                    df = df.drop(columns=['Date', 'Time', 'Sr No'], errors='ignore')
                    combined_df = pd.concat([combined_df, df], axis=1)

            # If the asset file is missing, fill with zeros using renamed column labels
            if not asset_found:
                missing_data = pd.DataFrame(0, index=range(len(combined_df)), columns=['tot_acc', 'ver_vel', 'ax_vel', 'hor_vel', 'temp', 'aud'])
                combined_df = pd.concat([combined_df, missing_data], axis=1)

        if not combined_df.empty:
            combined_df.fillna(0, inplace=True)
            combined_df['Code'] = ''
            combined_df.to_excel(test_file_path, index=False)
            st.success("All files processed and combined successfully!")
        else:
            st.error("No data found or processed. Please check files and asset names.")


    

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")


if st.button("Preprocess Files"):
    preprocess_files()






#################### Classification    ###############################





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
        df = pd.read_excel(file_path, sheet_name= 'Classification')        
        X = df.iloc[:, 3:-1].values  # Features (assuming columns 3 to second last)
        y = df['Code'].values  # Target column
        return X, y

    def preprocess_data(X, y):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Save the scaler
        joblib.dump(scaler, os.path.join(model_folder_path, "scaler_nn12345678912.pkl"))

        # Split into training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.01, random_state=42, shuffle=True)

        # Handle imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        return X_resampled, X_test, y_resampled, y_test

    def build_nn_model():
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(X_resampled.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
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
    joblib.dump(ensemble_model, os.path.join(model_folder_path, "nn_model12345678912.pkl"))
    st.success("Ensemble model training completed and saved!")

# Define the prediction function
def predict_ensemble(test_file_path, model_folder_path):
    def load_test_data(file_path):
        df = pd.read_excel(file_path)
        X_test = df.iloc[:, 3:-1].values
        return df, X_test

    def preprocess_test_data(X_test):
        scaler = joblib.load(os.path.join(model_folder_path, "scaler_nn12345678912.pkl"))
        X_test_scaled = scaler.transform(X_test)
        return X_test_scaled

    def predict(X_test_scaled):
        ensemble_model = joblib.load(os.path.join(model_folder_path, "nn_model12345678912.pkl"))
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



#...CHNAGED................................................................................................................................................

if st.button("Check BD Classification"):
    with st.spinner("Checking breakdown..."):
        #train_ensemble_model(training_file_path, model_folder_path)  # Train the model
        result = predict_ensemble(test_file_path, model_folder_path)  # Predict breakdown
        
        # Store the result in session state
        st.session_state["bd_output"] = result
        
        # Update session state based on the output
        if result != "No BD predicted":
            st.session_state["check_bd_clicked"] = True
        else:
            st.session_state["check_bd_clicked"] = False
    
    # Display the result
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
        joblib.dump(scaler, os.path.join(model_folder_path, 'scalerfinT111.pkl'))
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
    model.save(os.path.join(model_folder_path, 'trained_modelFINT111.h5'))

# Define the prediction function
def predict_time(test_file_path):
    def load_test_data(file_path):
        df = pd.read_excel(file_path)
        serial_numbers = df.iloc[:, 2].values
        times = df.iloc[:, 1].values
        X_test = df.iloc[:, 3:74].values
        return df, X_test, serial_numbers, times

    def preprocess_test_data(X_test):
        scaler = joblib.load(os.path.join(model_folder_path, 'scalerfinT111.pkl'))
        X_test_scaled = scaler.transform(X_test)
        return X_test_scaled

    def predict_time_to_breakdown(X_test_scaled):
        model = load_model(os.path.join(model_folder_path, 'trained_modelFINT111.h5'))
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
    #CHANGE.........................................................................................................................
    def find_minimum_maximum_and_mode_interval(time_to_breakdown_with_time):
        # Filter out negative times
        positive_times = [time for time in time_to_breakdown_with_time if time >= 0]
    
        if not positive_times:
            return None, None, None, None  # Handle no positive breakdown times case
    
        # Calculate minimum and maximum times
        min_time = min(positive_times)
        max_time = max(positive_times)
    
        # Define intervals of 5 units
        interval_start = min_time
        interval_end = max_time + 5  # Extend range to include the last value
        bins = []
        frequencies = []
    
        while interval_start < interval_end:
            # Create interval range
            interval = (interval_start, interval_start + 5)
            bins.append(interval)
    
            # Count occurrences within the interval
            frequency = sum(1 for time in positive_times if interval[0] <= time < interval[1])
            frequencies.append(frequency)
    
            # Move to the next interval
            interval_start += 5
    
        # Find the mode interval (highest frequency)
        max_frequency = max(frequencies)
        mode_index = frequencies.index(max_frequency)
        mode_interval = bins[mode_index]
    
        return min_time, max_time, mode_interval, max_frequency

   
    try:
        # Load and preprocess the test data
        df, X_test, serial_numbers, times = load_test_data(test_file_path)
        X_test_scaled = preprocess_test_data(X_test)
    
        # Make predictions
        predictions = predict_time_to_breakdown(X_test_scaled)
        predictions_with_time = calculate_time_difference(times, predictions)
    
        # Find the minimum, maximum, and mode interval
        min_time, max_time, mode_interval, mode_frequency = find_minimum_maximum_and_mode_interval(predictions_with_time)
    
        
    
       
    
        # Return the final weighted breakdown time
        # Choose the output based on the condition
        final_output_time = max(24,min(24 + (0.4*min_time + 0.6*max_time),48))
        

        


        # Return the final breakdown time
        return (f"Breakdown might occur in approximately w.r.t 6 AM Data date: \n"
        f"{final_output_time:.2f} hours")

    except Exception as e:
        return f"Error: {e}"
   

    


# Streamlit app UI
st.title("Time Prediction")


#....CHANGED........................................................................................................................................


if st.button("Predict Time", disabled=not st.session_state["check_bd_clicked"]):
    if st.session_state["bd_output"] == "No BD predicted":
         st.error("No breakdown predicted. Cannot proceed with time prediction.")
    else:
         with st.spinner("Training the model and making predictions..."):
             #train_model(training_file_path)
             result = predict_time(test_file_path)  # Predict time using predefined test data
         st.write(f"Predicted Time to Breakdown: {result}")
         st.success("Prediction complete!")

# # Streamlit app UI
# st.title("Time Prediction")

#if st.button("Predict Time"):
#   with st.spinner("Training the model and making predictions..."):
#       train_model(training_file_path)
#        result = predict_time(test_file_path)  # Predict time using predefined test data
#    st.write(f"Predicted Time to Breakdown: {result}")
 #   st.success("Prediction complete!")





    









#..........................................Trend..............................






import matplotlib.pyplot as plt

# Mapping for parameters to descriptive names
parameter_mapping = {
    'a2': 'Acceleration',
    'av2': 'Axial Velocity',
    'vv2': 'Vertical Velocity',
    'hv2': 'Horizontal Velocity',
    't2': 'Temperature',
    'd2': 'Audio'
}

# Column types with "All" option for UI
column_types_ui = ['All'] + list(parameter_mapping.values())

# Reverse mapping for internal logic
reverse_parameter_mapping = {v: k for k, v in parameter_mapping.items()}

# Streamlit UI
st.title("Trend Visualization for Sensor Data")

# Validate files
if not os.path.exists(test_file_path) or not os.path.exists(threshold_file_path):
    st.error("Required files not found! Ensure the test and threshold file paths are correct.")
else:
    try:
        # Load test and threshold data
        test_df = pd.read_excel(test_file_path)
        threshold_df = pd.read_excel(threshold_file_path)

        if test_df.empty:
            st.warning("NO DATA in the test file.")
        else:
            # Extract alternate sensor names
            sensor_mapping = threshold_df[['Asset name', 'Sensor name']].drop_duplicates()
            asset_to_sensor = dict(zip(sensor_mapping['Asset name'], sensor_mapping['Sensor name']))

            # UI filter with alternate names
            sensor_names = list(asset_to_sensor.values())
            selected_sensor_name = st.selectbox("Select the sensor", sensor_names, index=0)

            # Map selected sensor name back to the asset name
            selected_asset = next(asset for asset, sensor in asset_to_sensor.items() if sensor == selected_sensor_name)

            selected_column_ui = st.selectbox("Select parameter", column_types_ui, index=0)

            # Map the selected UI parameter back to its internal name
            if selected_column_ui == 'All':
                selected_column = 'All'
            else:
                selected_column = reverse_parameter_mapping[selected_column_ui]

            # Check if test data contains the required columns
            if selected_column == 'All':
                asset_columns = [f"{selected_asset}_{param}" for param in parameter_mapping.keys()]
            else:
                asset_columns = [f"{selected_asset}_{selected_column}"]

            if not all(col in test_df.columns for col in asset_columns):
                st.warning("Selected asset or columns not found in the test dataset.")
            else:
                # Extract relevant data for the selected asset and column type(s)
                time_data = test_df['Time']
                date_data = test_df['Date']
                datetime_data = pd.to_datetime(date_data + ' ' + time_data, format='%d-%m-%Y %I:%M %p')

                # Determine start and end dates for the X-axis label
                start_date = datetime_data.min().strftime('%d-%m-%Y')
                end_date = datetime_data.max().strftime('%d-%m-%Y')

                # Generate hourly tick labels
                hourly_ticks = pd.date_range(start=datetime_data.min(), end=datetime_data.max(), freq='H')

                # Prepare the plot
                plt.figure(figsize=(12, 6))

                if selected_column == 'All':
                    # Plot all parameters for the selected asset
                    for param, display_name in parameter_mapping.items():
                        column_name = f"{selected_asset}_{param}"
                        column_data = test_df[column_name]
                        plt.plot(datetime_data, column_data, linestyle='-', label=display_name)
                else:
                    # Plot the specific parameter
                    column_data = test_df[f"{selected_asset}_{selected_column}"]
                    plt.plot(datetime_data, column_data, linestyle='-', label=selected_column_ui)

                    # Get threshold values for the selected asset and parameter
                    threshold_row = threshold_df[
                        (threshold_df['Asset name'] == selected_asset) &
                        (threshold_df['Parameter'] == selected_column)
                    ]
                    if not threshold_row.empty:
                        caution_value = threshold_row['Caution'].values[0]
                        warning_value = threshold_row['Warning'].values[0]

                        # Add horizontal lines for caution and warning thresholds
                        plt.axhline(y=caution_value, color='orange', linestyle='--', label="Caution Threshold")
                        plt.axhline(y=warning_value, color='red', linestyle='--', label="Warning Threshold")

                # Configure the plot
                plt.xlabel(f"Time ({start_date} - {end_date})")
                plt.ylabel("Values")
                plt.title(f"Trend for {selected_sensor_name} - {selected_column_ui}")
                plt.xticks(hourly_ticks, [tick.strftime('%I %p') for tick in hourly_ticks], rotation=45)
                plt.grid(True)
                plt.legend(loc='upper left')  # Place the legend in the top-left corner
                plt.tight_layout()

                # Display the plot
                st.pyplot(plt)

                # Add functionality for threshold crossing counts
                warning_counts = {}
                caution_counts = {}

                for param, display_name in parameter_mapping.items():
                    column_name = f"{selected_asset}_{param}"
                    threshold_row = threshold_df[
                        (threshold_df['Asset name'] == selected_asset) &
                        (threshold_df['Parameter'] == param)
                    ]

                    if not threshold_row.empty:
                        caution_value = threshold_row['Caution'].values[0]
                        warning_value = threshold_row['Warning'].values[0]

                        # Count how many times the parameter crosses caution and warning thresholds
                        caution_counts[display_name] = (test_df[column_name] > caution_value).sum()
                        warning_counts[display_name] = (test_df[column_name] > warning_value).sum()
                    else:
                        caution_counts[display_name] = 0
                        warning_counts[display_name] = 0

                
                
                # Combine threshold crossing counts into a single table
                combined_df = pd.DataFrame(
                    {
                        "Parameter": list(parameter_mapping.values()),
                        "Caution Crossings": [caution_counts[display_name] for display_name in parameter_mapping.values()],
                        "Warning Crossings": [warning_counts[display_name] for display_name in parameter_mapping.values()]
                    }
                )
                
               # Create a new table with Sensor Name displayed only once
                sensor_row = pd.DataFrame({"Parameter": ["Sensor Name"], "Caution Crossings": [selected_sensor_name], "Warning Crossings": [""]})
                combined_df = pd.concat([sensor_row, combined_df], ignore_index=True)

                # Adjust the column names
                combined_df.columns = ["Parameter", "Caution Crossings", "Warning Crossings"]

                # Display the combined table

                st.markdown("### Threshold Crossing frequency")
                st.table(combined_df.T)

    except Exception as e:
        st.error(f"Error processing the files: {e}")















