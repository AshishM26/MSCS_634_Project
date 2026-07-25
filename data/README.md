# Dataset

This project uses only the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

`Project_Deliverable_1.ipynb` first uses a local copy and otherwise downloads the public dataset with:

```python
import kagglehub

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
```

The notebook copies the downloaded `creditcard.csv` here:

```text
data/creditcard.csv
```