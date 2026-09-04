import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Loading datasets...")

fake = pd.read_csv("fake.csv", low_memory=False)
true = pd.read_csv("true.csv", low_memory=False)

fake['label'] = 1
true['label'] = 0

fake = fake[['title', 'text', 'label']]
true = true[['title', 'text', 'label']]

df = pd.concat([fake, true], axis=0)
df = df.sample(frac=1).reset_index(drop=True)

df['content'] = df['title'] + " " + df['text']
df.drop(['title', 'text'], axis=1, inplace=True)
df.dropna(inplace=True)

print("Cleaning text...")

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return " ".join(text)

df['clean_content'] = df['content'].apply(clean_text)

print("Vectorizing text...")

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_content'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model...")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\n--- Model Evaluation ---")

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))