

# Financial Complaint Classifier

An NLP-based machine learning project that automatically classifies consumer financial complaints into 11 product categories using TF-IDF and Logistic Regression.

## Project Overview


This project uses TF-IDF feature extraction and Logistic Regression to classify complaint narratives from the Consumer Financial Protection Bureau (CFPB) Consumer Complaint Database.

The goal is to build a reproducible text-classification pipeline, evaluate its performance, and provide reusable training and prediction modules.

## Dataset

Source: CFPB Consumer Complaint Database

- Original records: 614,945
- Cleaned records: 19,163
- Target column: `Product`
- Input column: `Consumer complaint narrative`
- Number of classes: 11

### Data Cleaning

The cleaning pipeline:

1. Removes missing complaint narratives.
2. Removes narratives associated with conflicting product labels.
3. Removes exact duplicate narratives.
4. Removes normalized duplicate narratives.

The original narrative text is preserved for model training.

Raw and processed datasets are excluded from version control.

## Model Architecture

Complaint Text
    |
    v
TF-IDF Vectorizer
    |
    v
Logistic Regression
    |
    v
Predicted Product Category

Selected hyperparameters:

- min_df = 2
- ngram_range = (1, 2)
- class_weight = balanced
- C = 2.0
- max_iter = 1000

Hyperparameters were selected using validation experiments and stratified cross-validation.

## Final Model Performance

Evaluation on 3,833 held-out test complaints:

| Metric | Score |
|---|---|
| Accuracy | 81.55% |
| Macro F1-score | 73.80% |

The final model was trained using 15,330 complaints and evaluated on a separate held-out test set.



### Reproducing the Final Evaluation

After preparing the cleaned CFPB dataset, run:

```powershell
python -m src.evaluate
```

The script:

1. Creates a reproducible stratified 80/20 train-test split.
2. Trains the selected TF-IDF + Logistic Regression pipeline on 80% of the cleaned data.
3. Evaluates the model on the held-out 20%.
4. Saves aggregated evaluation results to `reports/test_metrics.json`.

The report includes:
- Test accuracy
- Macro F1-score
- Per-class precision, recall, F1-score, and support
- Confusion matrix and class labels

The selected hyperparameters were determined before final test evaluation.

Reported test results:

| Metric | Value |
|---|---|
| Test samples | 3,833 |
| Accuracy | 0.8155 |
| Macro F1-score | 0.7380 |

The report contains aggregated metrics only, not original complaint narratives.
```

### Limitations

- Performance varies across product categories.
- Debt or credit management and prepaid card complaints have relatively low recall.
- The dataset focuses on financial complaints rather than general customer support tickets.
- Model predictions should not replace human review in high-impact decisions.

## Project Structure

```text
data/          Local raw and processed datasets
notebooks/     EDA, preprocessing, baseline, and experiments
src/           Data processing, training, and prediction
tests/         Automated tests
models/        Locally saved trained models
requirements.txt
README.md
```

## Installation

Python 3.14 is recommended for reproducing the development environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```


## Dataset Preparation

This project uses the Consumer Financial Protection Bureau
(CFPB) Consumer Complaint Database.

Download the dataset from:
https://www.consumerfinance.gov/data-research/consumer-complaints/

The experiments in this repository used the dataset export:

`CCDB_Export_16_March_2026.csv`

Place the CSV file inside:

`data/raw/`

The training scripts expect a cleaned dataset at:

`data/processed/cleaned_complaints.csv`

To generate it, run the following command from the project root:

```powershell
python -c "import pandas as pd; from pathlib import Path; from src.data_processing import clean_dataset; source=Path('data/raw/CCDB_Export_16_March_2026.csv'); output=Path('data/processed/cleaned_complaints.csv'); output.parent.mkdir(parents=True,exist_ok=True); clean_dataset(pd.read_csv(source)).to_csv(output,index=False)"
```

The processed dataset contains 19,163 complaint narratives
when generated from the original export used in this project.

Different dataset versions may produce different results.

Neither the raw data nor the cleaned dataset is included
in this repository.
```



## Running Tests

```powershell
python -m pytest tests/ -v
```

## Training the Model

After obtaining and preparing the CFPB dataset:

```python
from src.train import train_final_model, save_model

model = train_final_model("data/processed/cleaned_complaints.csv")
save_model(model, "models/ticket_classifier.joblib")
```

## Making Predictions

```python
from src.predict import load_model, predict_complaint

model = load_model("models/ticket_classifier.joblib")

prediction = predict_complaint(
    model,
    "My credit card payment was charged twice."
)

print(prediction)
```

## Reproducibility

- Fixed random seed: 42
- Stratified train/validation/test splits
- TF-IDF and Logistic Regression combined in a Pipeline
- Held-out test evaluation
- Automated tests using pytest

## Future Improvements

- Improve minority-class recall.
- Compare alternative text classifiers.
- Add a REST API for inference.
- Add CI/CD workflows.
- Monitor model performance on new complaints.

## License

No license has been selected yet.
