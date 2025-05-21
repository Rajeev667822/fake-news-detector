import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from wordcloud import WordCloud  

# Load datasets  
fake_df = pd.read_csv("fake.csv")  
real_df = pd.read_csv("Real.csv")  

# Add labels  
fake_df["label"] = "Fake"  
real_df["label"] = "Real"  

# Merge datasets  
df = pd.concat([fake_df, real_df], ignore_index=True)  


sns.countplot(data=df, x="label", palette="coolwarm")  
# plt.title("Fake vs Real News Count")  
# plt.xlabel("News Type")  
# plt.ylabel("Count")  
# plt.show()

# #word count
# df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))  

# sns.boxplot(x="label", y="word_count", data=df, palette="coolwarm")  
# plt.title("Word Count Distribution in Fake and Real News")  
# plt.show()


#generate word cloud real and fake


# fake_text = " ".join(fake_df["text"].dropna())  
# real_text = " ".join(real_df["text"].dropna())  

# # Fake news word cloud  
# plt.figure(figsize=(10, 5))  
# plt.imshow(WordCloud(width=800, height=400).generate(fake_text))  
# plt.axis("off")  
# plt.title("Fake News Word Cloud")  
# plt.show()  

# # Real news word cloud  
# plt.figure(figsize=(10, 5))  
# plt.imshow(WordCloud(width=800, height=400).generate(real_text) )  
# plt.axis("off")  
# plt.title("Real News Word Cloud")  
# plt.show()


# from sklearn.feature_extraction.text import TfidfVectorizer  

# vectorizer = TfidfVectorizer(stop_words="english", max_features=20)  
# X = vectorizer.fit_transform(df["text"].dropna())  
# tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())  

# plt.figure(figsize=(10, 5))  
# sns.barplot(x=tfidf_df.mean().sort_values(ascending=False)[:10].index,  
#             y=tfidf_df.mean().sort_values(ascending=False)[:10].values,  
#             palette="coolwarm")  
# plt.xticks(rotation=45)  
# plt.title("Top 10 Common Words (TF-IDF)")  
# plt.show()
