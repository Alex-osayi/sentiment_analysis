import spacy
from spacytextblob.spacytextblob import SpacyTextBlob  # for sentiment analysis
import pandas as pd

# Load the spaCy model and add the SpacyTextBlob extension
nlp = spacy.load('en_core_web_md')
nlp.add_pipe('spacytextblob')

# Read the CSV file containing Amazon product reviews into a DataFrame
dataframe = pd.read_csv('amazon_product_reviews.csv')

# Drop rows where the 'reviews.text' column is null
clean_data = dataframe.dropna(subset=['reviews.text'])

def calculate_polarity(doc):
    """
    Calculate the polarity of a given spaCy doc.
    
    Parameters:
        doc (spacy.Doc): The spaCy doc representing a text.
        
    Returns:
        float: The polarity score of the text.
    """
    return doc._.blob.polarity

def extract_sentiment(doc):
    """
    Extract the sentiment of a given spaCy doc.
    
    Parameters:
        doc (spacy.Doc): The spaCy doc representing a text.
        
    Returns:
        tuple: A tuple containing sentiment values (polarity, subjectivity).
    """
    return doc._.blob.sentiment

def determine_sentiment_label(polarity):
    """
    Determine the sentiment label based on the polarity score.
    
    Parameters:
        polarity (float): The polarity score of the text.
        
    Returns:
        str: The sentiment label ('Positive', 'Negative', or 'Neutral').
    """
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def analyze_sentiment(review):
    """
    Analyze the sentiment of a given review text.
    
    Parameters:
        review (str): The text of the review to be analyzed.
        
    Returns:
        tuple: A tuple containing polarity, sentiment, and sentiment label.
    """
    doc = nlp(review)
    polarity = calculate_polarity(doc)
    sentiment = extract_sentiment(doc)
    sentiment_label = determine_sentiment_label(polarity)
    
    return polarity, sentiment, sentiment_label

# Define the row indices to analyze
row_indices = [4, 7]

# Apply sentiment analysis function to specified rows
for row_index in row_indices:
    sample_review = clean_data['reviews.text'].iloc[row_index]
    polarity, sentiment, sentiment_label = analyze_sentiment(sample_review)

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
