import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string

# If you're running this for the first time, you'll need to download the stopwords package
nltk.download('punkt')
nltk.download('stopwords')

# Load the data from CSV file
df = pd.read_csv('abc.csv')

# Choose the 8th column (index 7)
text_data = df.iloc[:,7]

# Convert text to lowercase
text_data = text_data.str.lower()

# Remove punctuation
text_data = text_data.str.translate(str.maketrans('', '', string.punctuation))

# Convert all data to string type
text_data = text_data.astype(str)

# Remove stopwords
stop_words = set(stopwords.words('english'))

# Add custom stop words
#Palavras que não fazem sentido para a análise
custom_stop_words = ['de', 'e', 'para','nan','que','não','em','estava','eu','um','pra','sem']  # replace with your words
stop_words = stop_words.union(custom_stop_words)

# Word tokenize the text
text_data = text_data.apply(word_tokenize)

# Remove stop words
text_data = text_data.apply(lambda x: [word for word in x if word not in stop_words])

# Combine all the words into a single list
all_words = text_data.sum()

# Count the occurrences of each word
word_counts = Counter(all_words)

# Get the top 10 most common words
top_10_words = word_counts.most_common(10)

# Convert list of tuples to dictionary
top_10_dict = dict(top_10_words)

# Generate a word cloud image
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_10_dict)

# Display the generated image with matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
