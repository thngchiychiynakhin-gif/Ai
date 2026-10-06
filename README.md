# OrchardGuard AI

Flask API and Random Forest training code for OrchardGuard.

The sample datasets are synthetic demonstration data, not field measurements.

## Run the API

Install Flask, joblib, NumPy, pandas, and scikit-learn, then run:

```powershell
python .\ai_api.py
```

The API loads `orchardguard_model.pkl` and expects soil moisture as a percentage
from 0 to 100.

## Train the model

From this directory, run:

```powershell
python .\train_model.py
```

The training script reads the default CSV in
`OrchardGuard-AI\dataset\orchard_dataset.csv`, converts the 12-bit soil ADC
values to percentages, backs up an existing model, and writes a new
`orchardguard_model.pkl`.
