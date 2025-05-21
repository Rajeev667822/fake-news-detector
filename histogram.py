import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
real_news = pd.read_csv("Real.csv")
fake_news = pd.read_csv("fake.csv")

# Drop missing values
real_news.dropna(inplace=True)
fake_news.dropna(inplace=True)

# Function to calculate text lengths
def text_length(text):
    return len(str(text).split())

# Apply text_length function
real_news['text_length'] = real_news['text'].apply(text_length)
fake_news['text_length'] = fake_news['text'].apply(text_length)

# Plot barchart
# plt.figure(figsize=(10, 5))
plt.bar(real_news['text_length'], label="Real News", color="blue")
plt.bar(fake_news['text_length'], label="Fake News", color="red")

# Labels and Title
plt.xlabel("Number of Words in Article")
plt.ylabel("Frequency")
plt.title("Distribution of Word Count in Fake vs. Real News")
plt.legend()
plt.show()
