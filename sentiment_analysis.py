import spacy
from spacytextblob.spacytextblob import SpacyTextBlob  # for sentiment analysis
import pandas as pd

# Load the model and add the textblob extension
nlp = spacy.load('en_core_web_md')
nlp.add_pipe('spacytextblob')

dataframe = pd.read_csv('amazon_product_reviews.csv')

# Ensure that 'reviews.text' is not null
clean_data = dataframe.dropna(subset=['reviews.text'])

# Function for sentiment analysis
def analyze_sentiment(review):
    doc = nlp(review)
    
    # Using the polarity attribute
    polarity = doc._.blob.polarity
    
    # Using the sentiment attribute
    sentiment = doc._.blob.sentiment
    
    return polarity, sentiment

# Define the row indices to analyze
row_indices = [4, 7]  # Index of the rows to analyze

# Apply sentiment analysis function to specified rows
for row_index in row_indices:
    sample_review = clean_data['reviews.text'].iloc[row_index]
    polarity, sentiment = analyze_sentiment(sample_review)

    # Add a column for sentiment labels
    sentiment_label = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

    # Display the results for the current row
    print(f"Row {row_index + 1}:")
    print(f"Review: {sample_review}")
    print(f"Polarity: {polarity}")
    print(f"Sentiment: {sentiment}")
    print(f"Sentiment Label: {sentiment_label}")
    print("\n")

# Select the reviews for rows 5 and 8
review_5 = clean_data['reviews.text'].iloc[4]
review_8 = clean_data['reviews.text'].iloc[7]

# Process the text using spaCy
doc_5 = nlp(review_5)
doc_8 = nlp(review_8)

# Calculate the similarity between the two reviews
similarity_score = doc_5.similarity(doc_8)

# Print the similarity score
print(f"Similarity between row 5 and row 8: {similarity_score}")
