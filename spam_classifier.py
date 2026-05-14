import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import sklearn.metrics

print("All done!")

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', names=['label', 'message'])
print(df)

print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(df['label'].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

sns.countplot(x='label', data=df)
plt.title("Spam vs Ham Distribution")
plt.show()

X = df['message']
y = df['label']

vectorizer = TfidfVectorizer(stop_words='english')

X_transformed = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y,
    test_size=0.3,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(sklearn.metrics.classification_report(y_test, y_pred))
sample = "Congratulations! You won a free ticket" # @param {type:"string"}

sample_transformed = vectorizer.transform([sample])
prediction = model.predict(sample_transformed)

print(f"Message: {sample}")
print(f"Prediction: {prediction[0]}")
