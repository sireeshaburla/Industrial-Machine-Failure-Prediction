# Industrial Machine Failure Prediction

## Objective
Predict whether an industrial machine is likely to experience a failure using operating and maintenance indicators.

## Features
- temperature
- vibration
- pressure
- runtime
- maintenance_gap

## Target
`failure`
- 0 = No failure risk
- 1 = Failure risk

## Dataset
`dataset.csv` in this repository is a **synthetic educational dataset generated for this capstone prototype**. It is not real industrial sensor data.

## Model
The project uses Logistic Regression with:
1. median imputation
2. standard scaling
3. Logistic Regression

The preprocessing and model are saved together as `model.pkl`.

## Run locally

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Project files
- `dataset.csv` — dataset
- `machine_failure.ipynb` — analysis notebook
- `train.py` — model training/evaluation
- `app.py` — Streamlit application
- `model.pkl` — trained pipeline
- `requirements.txt` — dependencies
- `README.md` — project documentation
- `technical_paper.pdf` — report
- `presentation.pptx` — presentation

## Limitations
This is an educational prototype using synthetic data. It should not be used for real industrial maintenance or safety decisions.
