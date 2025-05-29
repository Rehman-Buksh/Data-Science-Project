
import streamlit as st
import pandas as pd
import pickle
import json

# Load model and encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

with open("columns.json", "r") as f:
    encoded_columns = json.load(f)

st.title("Energy Usage Predictor")

# Text Inputs
Filename = st.text_input("Filename")
Platform = st.text_input("Platform")
Architecture = st.text_input("Architecture")
Processor = st.text_input("Processor")
ProcessorGen = st.text_input("ProcessorGen")
Compiler = st.text_input("Compiler")
ParallelizationTech = st.text_input("ParallelizationTech")

# Numeric Inputs
Optimization = st.number_input("Optimization", value=0)
ThreadsUsed = st.number_input("ThreadsUsed", value=1)
Runtime = st.number_input("Runtime", value=0.0)
EnergyCores = st.number_input("EnergyCores", value=0.0)
EnergyPkg = st.number_input("EnergyPkg", value=0.0)
EnergyMem = st.number_input("EnergyMem", value=0.0)

# Combine inputs into a DataFrame
input_df = pd.DataFrame({
    'Filename': [Filename],
    'Platform': [Platform],
    'Architecture': [Architecture],
    'Processor': [Processor],
    'ProcessorGen': [ProcessorGen],
    'Compiler': [Compiler],
    'ParallelizationTech': [ParallelizationTech],
    'Optimization': [Optimization],
    'ThreadsUsed': [ThreadsUsed],
    'Runtime': [Runtime],
    'EnergyCores': [EnergyCores],
    'EnergyPkg': [EnergyPkg],
    'EnergyMem': [EnergyMem]
})

# Encode
cat_cols = ['Filename', 'Platform', 'Architecture', 'Processor', 'ProcessorGen', 'Compiler', 'ParallelizationTech']
num_cols = ['Optimization', 'ThreadsUsed', 'Runtime', 'EnergyCores', 'EnergyPkg', 'EnergyMem']
encoded_cat = encoder.transform(input_df[cat_cols])
final_input = pd.concat([encoded_cat, input_df[num_cols].reset_index(drop=True)], axis=1)

# Ensure all required columns are present
for col in encoded_columns:
    if col not in final_input.columns:
        final_input[col] = 0
final_input = final_input[encoded_columns]

# Predict
if st.button("Predict Total Energy"):
    prediction = model.predict(final_input)
    st.success(f"Predicted Total Energy: {prediction[0]:.2f}")
