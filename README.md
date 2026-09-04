# Fake News Detection

A machine learning project that detects whether a news article is fake or true using Natural Language Processing (NLP).

## Technologies Used

- Python
- Pandas
- NLTK
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression

## How It Works

1. Loads fake and true news datasets.
2. Combines the news title and text.
3. Preprocesses the text using:
   - Lowercasing
   - Removing non-alphabetic characters
   - Removing stopwords
   - Stemming using Porter Stemmer
4. Converts text into numerical features using TF-IDF.
5. Splits the dataset into training and testing data.
6. Trains a Logistic Regression model.
7. Evaluates the model using accuracy, confusion matrix, and classification report.

## How to Run

Install the required libraries:

```bash
pip install pandas nltk scikit-learn
