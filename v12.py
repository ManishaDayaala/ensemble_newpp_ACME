
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
#uploaded_files = st.file_uploader("Upload Excel files", type=['xlsx'], accept_multiple_files=True)

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
    required_column_indices = [5, 8, 11, 14, 17]  # 0-based indices for F, I, L, O, R, U
    required_column_names = ['a2', 'vv2', 'av2', 'hv2', 't2']

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

            # Ensure all required columns exist and fill missing ones with zeros
            for col in required_column_names:
                if col not in data_for_asset.columns:
                    data_for_asset[col] = 0

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
    date_list = [(start_time + timedelta(minutes=30 * i)).strftime('%d %b %Y') for i in range(49)]
    time_list = [(start_time + timedelta(minutes=30 * i)).strftime('%I:%M %p') for i in range(49)]
    sr_no_list = list(range(1, 50))

    # Insert Date, Time, and Sr No columns into the final output DataFrame
    output_df.insert(0, 'Date', date_list)
    output_df.insert(1, 'Time', time_list)
    output_df.insert(2, 'Sr No', sr_no_list)

    # Add an empty 'Code' column at the end
    output_df['Code'] = 0

    # Fill NaN values in the DataFrame with 0
    output_df = output_df.fillna(0)


    # Save the processed data using ExcelWriter
    with pd.ExcelWriter(test_file_path, engine='openpyxl') as writer:
        output_df.to_excel(writer, index=False)

    # Display success message when all files are processed
    st.info(f"Data has been processed and saved")

# Create a button to trigger the process
if st.button('Preprocess Data'):
    process_data()




#################### Classification    ###############################

# 🚀 Install if not already done:
# pip install pandas numpy scikit-learn xgboost imbalanced-learn tensorflow joblib streamlit openpyxl

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape

# ---------------------
# ⚙️ Set random seed
# ---------------------
def set_random_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

# ---------------------
# 📦 Training Function
# ---------------------
def train_ensemble_model_shifted_label(training_file_path, model_folder_path):
    set_random_seed(seed=42)

    df = pd.read_excel(training_file_path, sheet_name='Classification')
    df['Code'] = df['Code'].shift(-48)  # Shift label 1 day ahead (48 half-hour intervals)
    df.dropna(inplace=True)

    X = df.iloc[:, 1:-1].values  # Sensor features
    y = df['Code'].astype(int).values  # Shifted labels

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(model_folder_path, "scaler_shifted1234.pkl"))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


    
    # Sample weights for XGBoost
    class_weights = {0: 1.0, 1: 30, 2: 50, 3: 30}
    sample_weights = np.array([class_weights[int(label)] for label in y_resampled])

    

    xgb_model = XGBClassifier(
        n_estimators=400,
        objective='multi:softmax',
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        num_class=4
    )

   
    def build_nn_model():
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(X_resampled.shape[1],)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(4, activation='softmax'))  # 4 output classes

        # Define class-specific alpha weights (adjust if needed)
        alpha_weights = [1.0, 20.0, 20.0, 20.0]  # Code 0 gets less weight
    
        # Compile with focal loss
        model.compile(
            optimizer='adam',
            loss=focal_loss(gamma=2., alpha=alpha_weights),
            metrics=['accuracy']
        )
        return model


    keras_model = KerasClassifier(
        model=build_nn_model,
        epochs=100,
        batch_size=32,
        verbose=0,
        #class_weight=class_weights
    )

    # Ensemble
    ensemble = VotingClassifier(
        estimators=[
        ('xgb', xgb_model), 
        ('lstm', keras_model)],
        voting='soft'
    )

    ensemble.fit(X_resampled, y_resampled)
    joblib.dump(ensemble, os.path.join(model_folder_path, "ensemble_shifted_model1234.pkl"))
    st.success("✅ Training completed and model saved!")

# ---------------------
# 🔮 Prediction Function
# ---------------------
def predict_future_breakdown(test_file_path, model_folder_path):
    set_random_seed(seed=42)
    df = pd.read_excel(test_file_path)
    X = df.iloc[:, 3:-1].values  # All features

    scaler = joblib.load(os.path.join(model_folder_path, "scaler_shifted1234.pkl"))
    X_scaled = scaler.transform(X)

    model = joblib.load(os.path.join(model_folder_path, "ensemble_shifted_model1234.pkl"))
    preds = model.predict(X_scaled)

    labels = ["Code 0", "Code 1", "Code 2", "Code 3"]
    result_labels = [labels[p] for p in preds]

    non_zero = [lbl for lbl in result_labels if lbl != "Code 0"]
    if non_zero:
        return f"🚨 Potential Breakdown(s): {', '.join(set(non_zero))}"
    else:
        return "✅ No BD predicted"



# ---------------------
# 🌐 Streamlit UI
# ---------------------
st.title("🔮 Predict Future Breakdown (24hr Ahead)")

#if st.button("Train Model"):
#    if training_file_path:
#        with st.spinner("Training ensemble model..."):
#            train_ensemble_model_shifted_label(training_file_path, model_folder_path)
#    else:
#        st.warning("Please upload the training file!")

if st.button("Predict Future Breakdown"):
    if test_file_path:
        with st.spinner("Predicting..."):
            result = predict_future_breakdown(test_file_path, model_folder_path)
            st.subheader("🔍 Result:")
            st.write(result)
            # Store the result in session state
            st.session_state["bd_output"] = result
            
            # Update session state based on the output
            if result != "No BD predicted":
                st.session_state["check_bd_clicked"] = True
            else:
                st.session_state["check_bd_clicked"] = False
    else:
        st.warning("Please upload today's data for prediction.")


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
        X = df.iloc[:, 1:-1].values
        y = df.iloc[:, -1].values
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
        joblib.dump(scaler, os.path.join(model_folder_path, 'scaler_time_changed.pkl'))
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
    model.save(os.path.join(model_folder_path, 'trained_time_changed.h5'))

# Define the prediction function
def predict_time(test_file_path):
    def load_test_data(file_path):
        df = pd.read_excel(file_path)
        serial_numbers = df.iloc[:, 2].values
        times = df.iloc[:, 1].values
        X_test = df.iloc[:, 3:-1].values
        return df, X_test, serial_numbers, times

    def preprocess_test_data(X_test):
        scaler = joblib.load(os.path.join(model_folder_path, 'scaler_time_changed.pkl'))
        X_test_scaled = scaler.transform(X_test)
        return X_test_scaled

    def predict_time_to_breakdown(X_test_scaled):
        model = load_model(os.path.join(model_folder_path, 'trained_time_changed.h5'))
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
    
        
    
       
    
        final_output_time = 0.4 * min_time + 0.6 * max_time


        


        # Return the final breakdown time
        return (f"Breakdown might occur in approximately w.r.t 6 AM Data date: \n"
        f"{final_output_time:.2f} hours")

    except Exception as e:
        return f"Error: {e}"
   

    


# Streamlit app UI
st.title("Time Prediction")


#....CHANGED........................................................................................................................................


if st.button(("Predict Time"), disabled=not st.session_state["check_bd_clicked"]):
    if st.session_state["bd_output"] == "No BD predicted":
         st.error("No breakdown predicted. Cannot proceed with time prediction.")
    else:
         with st.spinner("Training the model and making predictions..."):
             #train_model(training_file_path)
             result = predict_time(test_file_path)  # Predict time using predefined test data
         st.write(f"Predicted Time to Breakdown: {result}")
         st.success("Prediction complete!")






##########################  LSTM Autoencoder for Anomaly Detection ###############################

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

# Set random seed
def set_random_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Function to train the LSTM autoencoder model
def train_lstm_autoencoder_model(training_file_path, model_folder_path):
    set_random_seed()

    # Load and preprocess training data
    df = pd.read_excel(training_file_path)
    #df = df.dropna()

    
    column_names_train = df.columns[1:-1]
    X = df[[col for col in column_names_train if not col.endswith(('_d2', '_t2'))]]

    # Normalize data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler
    joblib.dump(scaler, os.path.join(model_folder_path, "lstm_auto_scaler1.pkl"))

    # Reshape to 3D for LSTM (samples, time_steps, features)
    TIME_STEPS = 10
    def create_sequences(data, time_steps=TIME_STEPS):
        seqs = []
        for i in range(len(data) - time_steps):
            seqs.append(data[i:i + time_steps])
        return np.array(seqs)
    
    X_seq = create_sequences(X_scaled)

    # Define LSTM Autoencoder model
    inputs = Input(shape=(TIME_STEPS, X_seq.shape[2]))
    encoded = LSTM(64, activation="relu", return_sequences=False)(inputs)
    decoded = RepeatVector(TIME_STEPS)(encoded)
    decoded = LSTM(64, activation="relu", return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(X_seq.shape[2]))(decoded)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())

    # Train the model
    autoencoder.fit(X_seq, X_seq, epochs=20, batch_size=64, shuffle=True)

    # Save model
    autoencoder.save(os.path.join(model_folder_path, "lstm_auto_model1.h5"))                    

    st.success("LSTM Autoencoder training completed and model saved!")

def predict_lstm_autoencoder(test_file_path, model_folder_path):
    set_random_seed()
    import pandas as pd
    # Load test data
    df_test = pd.read_excel(test_file_path)

    column_names = df_test.columns[3:-1]
    X_test = df_test[[col for col in column_names if not col.endswith(('_d2', '_t2'))]]

    # Load scaler and scale
    scaler = joblib.load(os.path.join(model_folder_path, "lstm_auto_scaler1.pkl"))
    X_test_scaled = scaler.transform(X_test)

    # Create sequences
    TIME_STEPS = 10
    def create_sequences(data, time_steps=TIME_STEPS):
        seqs = []
        for i in range(len(data) - time_steps):
            seqs.append(data[i:i + time_steps])
        return np.array(seqs)

    X_test_seq = create_sequences(X_test_scaled)

    # Load model
    model = load_model(os.path.join(model_folder_path, "lstm_auto_model1.h5"))

    # Predict and calculate reconstruction error per feature
    X_pred = model.predict(X_test_seq)
    errors = np.mean(np.abs(X_pred - X_test_seq), axis=1)  # shape: (samples, features)

    # Threshold (e.g., 95th percentile of average feature-wise errors)
    overall_mae = np.mean(errors, axis=1)
    threshold = np.percentile(overall_mae, 95)

    anomaly_indices = np.where(overall_mae > threshold)[0]
    feature_anomalies = np.where(errors[anomaly_indices] > np.percentile(errors, 95), 1, 0)

    sensor_dict = {}

    for idx, feature_row in enumerate(feature_anomalies):
        for f_idx, is_anomaly in enumerate(feature_row):
            if is_anomaly:
                feature_name = column_names[f_idx]
                sensor_id = feature_name.split('_')[0]  # e.g., "Sensor3" from "Sensor3_Temperature"
                if sensor_id not in sensor_dict:
                    sensor_dict[sensor_id] = {"count": 0, "params": set()}
                sensor_dict[sensor_id]["count"] += 1
                sensor_dict[sensor_id]["params"].add(feature_name)

    # Filter only sensors with anomaly count > 30
    filtered_dict = {sensor: info for sensor, info in sensor_dict.items() if info["count"] > 15}

    if not filtered_dict:
        st.session_state["check_bd_clicked"] = False
        st.success("✅ No abnormalities detected in any sensor.")
        return "✅ No abnormalities detected in any sensor."

    # Display in table format
    import pandas as pd
    table_data = {
        "Sensor Having Abnormality": [],
        "Affected Parameters": [],
        "Anomaly Count": []
    }

    for sensor, info in filtered_dict.items():
        table_data["Sensor Having Abnormality"].append(sensor)
        table_data["Affected Parameters"].append(", ".join(info["params"]))
        table_data["Anomaly Count"].append(info["count"])

    df_result = pd.DataFrame(table_data)

    st.session_state["check_bd_clicked"] = True
    st.warning("🚨 Abnormalities Detected")
    st.dataframe(df_result, use_container_width=True)

    return df_result

# Streamlit app UI
st.title("Anamoly Detector")


# Inside Streamlit UI
if st.button("Check abnormality in sensors"):
    with st.spinner("🔍 Checking for abnormality..."):
        #train_lstm_autoencoder_model(training_file_path, model_folder_path)
        result = predict_lstm_autoencoder(test_file_path, model_folder_path)
        st.session_state["Anamoly_output"] = result

        if isinstance(result, str):  # Means no anomaly case
            st.markdown(f"```text\n{result}\n```")
        else:
            st.success("✅ Anomaly detection complete!")










    









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
                datetime_data = pd.to_datetime(date_data + ' ' + time_data, format='%d %b %Y %I:%M %p')

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















