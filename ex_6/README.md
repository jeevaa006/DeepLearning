# LSTM Sentiment Analysis on IMDB Movie Reviews

**Experiment No:** 6 – Implementation of LSTM for Sentiment Analysis  
**Department:** AI & DS

## Objective
To implement a Long Short-Term Memory (LSTM) network for sentiment analysis by performing text preprocessing, tokenization, sequence generation and embedding, and to classify movie reviews as **positive** or **negative**.

## Dataset
IMDB Movie Reviews (Stanford aclImdb, 50,000 labelled reviews, balanced classes).  
To keep training fast in Google Colab, a random subset is used:

| Set | Reviews |
|---|---|
| Training (80% of 20,000) | 16,000 |
| Validation (20% of 20,000) | 4,000 |
| Test (separate, unseen) | 5,000 |

## Technologies Used
Python, TensorFlow/Keras, NumPy, Pandas, Matplotlib, Scikit-learn, NLTK, Google Colab, GitHub.

## Methodology
1. **Part A – Preprocessing:** lowercase → remove HTML tags/punctuation/special characters → remove stop words (negations such as *not* and *no* are kept) → tokenize → build a 10,000-word vocabulary from training reviews only → convert to integer sequences → pad to 200 tokens → encode labels (negative = 0, positive = 1) → train/validation/test split.
2. **Part B – Embedding:** Keras `Embedding` layer (vocabulary 10,000, dimension 64) converts padded sequences of shape `(batch, 200)` into dense vectors of shape `(batch, 200, 64)`.
3. **Part C – Model:** build and compile the LSTM classifier.
4. **Part D – Training and prediction:** train with validation and early stopping, plot accuracy/loss, evaluate on the test set, predict sample reviews.

## Model Architecture
```
Input (200) -> Embedding (10000 x 64) -> LSTM (64 units) -> Dense (1, sigmoid)
```
Optimizer: Adam (lr = 0.001) | Loss: binary cross-entropy | Metric: accuracy  
Parameters: 673,089 (all trainable) — Embedding 640,000 + LSTM 33,024 + Dense 65.

## Evaluation Metrics
Accuracy, Precision, Recall, F1-score, Confusion Matrix (positive class = 1).

## Sample Results
> Fill these in from **your own** notebook run. Do not copy numbers from anywhere else.

| Metric | Value |
|---|---|
| Final training accuracy | _from your run_ |
| Final validation accuracy | _from your run_ |
| Test accuracy | _from your run_ |
| Precision | _from your run_ |
| Recall | _from your run_ |
| F1-score | _from your run_ |

Screenshots are in the `screenshots/` folder; sample predictions are in `results/sample_predictions.csv`.

## How to Run the Notebook
1. Open [Google Colab](https://colab.research.google.com) → **File → Upload notebook** → select `lstm_sentiment_analysis.ipynb`.
2. (Optional, faster) **Runtime → Change runtime type → T4 GPU**.
3. Run the cells from top to bottom (**Runtime → Run all**). The dataset is downloaded automatically.
4. Download `results/sample_predictions.csv` from the Colab **Files** panel and take the screenshots listed in the lab record.

To run locally: `pip install -r requirements.txt`, then open the notebook in Jupyter.

## GitHub Repository Structure
```
lstm-sentiment-analysis/
├── README.md
├── lstm_sentiment_analysis.ipynb
├── requirements.txt
├── screenshots/
│   ├── part-a-code.png, part-a-output.png
│   ├── part-b-code.png, part-b-output.png
│   ├── part-c-code.png, part-c-output.png
│   ├── part-d-code.png, part-d-output.png
│   ├── accuracy-plot.png, loss-plot.png
│   └── confusion-matrix.png
└── results/
    └── sample_predictions.csv
```

## Author
**Jeevaa**  
Roll No: 24BAD047
