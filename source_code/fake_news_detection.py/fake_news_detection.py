import pandas as pd

fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

print(fake.head())
print(real.head())
fake["label"] = 0
real["label"] = 1
data = pd.concat([fake, real])
print(data.shape)
data = data.drop_duplicates()
data = data.dropna()
x = data["text"]
y = data["label"]
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

x = vectorizer.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
from sklearn.linear_model import PassiveAggressiveClassifier

model = PassiveAggressiveClassifier()

model.fit(x_train, y_train)
from sklearn.metrics import accuracy_score

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

fake = pd.read_csv("dataset/Fake.csv")

text = " ".join(fake["text"])

wordcloud = WordCloud().generate(text)

plt.imshow(wordcloud)
plt.axis("off")
plt.show()
import matplotlib.pyplot as plt

data["label"].value_counts().plot(kind="bar")

plt.title("Fake vs Real News")

plt.show()