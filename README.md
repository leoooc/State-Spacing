# Mamba State-Space Model for UCI Air Quality Assessment

This repository demonstrates a custom state-space–inspired model, called the MambaModel, that uses a custom MambaLayer to learn state dynamics over time. It is designed as an alternative to more complicated S4-based layers and serves as a proof-of-concept for using learnable state transitions in a deep learning model.

# Overview

The project uses the UCI Air Quality dataset preprocessed into preprocessed_airquality.csv. A sliding-window dataset is created from this CSV, and the MambaModel is trained to predict the next time step’s feature vector based on past inputs.

The MambaLayer implements a simple state update mechanism:
	•	It learns a state transition matrix A and an input coupling matrix B.
	•	An initial state is learned and updated over the sequence using a nonlinearity.
	•	The final state is projected to form the output.

The MambaModel stacks multiple MambaLayers (with dropout between layers) to form a deep state-space–inspired architecture.

## Repository Structure

state-spaces/
├── preprocessed_airquality.csv
├── train_custom_mamba.py
├── README.md
├── requirements.txt
└── venv/          (optional)

# Requirements
	•	Python 3.8 or later
	•	PyTorch
	•	scikit-learn
	•	pandas
	•	numpy

 pip install -r requirements.txt

 # How to Run
	1.	Preprocess the Data:
Ensure that the UCI Air Quality dataset has been preprocessed into preprocessed_airquality.csv. (If you haven’t already, run the provided preprocess_uci.py script.)

	2.	Train the Model:
Run the training script:
python3 train_custom.py

The script will:
	•	Load and normalize the dataset.
	•	Create a sliding-window dataset for training, validation, and testing.
	•	Instantiate the MambaModel.
	•	Train the model for a specified number of epochs.
	•	Report training, validation, and test losses.
