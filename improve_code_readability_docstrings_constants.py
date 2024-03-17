# Constants
CSV_FILE_PATH = 'amazon_product_reviews.csv'
REVIEW_COLUMN_NAME = 'reviews.text'
POSITIVE_LABEL = "Positive"
NEGATIVE_LABEL = "Negative"
NEUTRAL_LABEL = "Neutral"

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd

# Load the model and add the textblob extension
nlp = spacy.load('en_core_web_md')
nlp.add_pipe('spacytextblob')

def load_data():
    """
    Load data from CSV file.
    
    Returns:
        pandas DataFrame: Data loaded from the CSV file.
    """
    try:
        return pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: File '{CSV_FILE_PATH}' not found.")
        return None
    except IOError:
        print(f"Error: Unable to read file '{CSV_FILE_PATH}'.")
        return None

def analyze_sentiment(review):
    """
    Perform sentiment analysis on the given review.
    
    Args:
        review (str): The text of the review.
    
    Returns:
        tuple: A tuple containing the polarity and sentiment label.
    """
    doc = nlp(review)
    polarity = doc._.blob.polarity
    sentiment_label = POSITIVE_LABEL if polarity > 0 else NEGATIVE_LABEL if polarity < 0 else NEUTRAL_LABEL
    return polarity, sentiment_label

# Load the data
dataframe = load_data()

if dataframe is not None:
    # Ensure that 'reviews.text' is not null
    clean_data = dataframe.dropna(subset=[REVIEW_COLUMN_NAME])

    # Define the row indices to analyze
    row_indices = [4, 7]  # Index of the rows to analyze

    # Apply sentiment analysis function to specified rows
    for row_index in row_indices:
        sample_review = clean_data[REVIEW_COLUMN_NAME].iloc[row_index]
        polarity, sentiment_label = analyze_sentiment(sample_review)

        # Display the results for the current row
        print(f"Row {row_index + 1}:")
        print(f"Review: {sample_review}")
        print(f"Polarity: {polarity}")
        print(f"Sentiment Label: {sentiment_label}")
        print("\n")

    # Select the reviews for rows 5 and 8
    review_5 = clean_data[REVIEW_COLUMN_NAME].iloc[4]
    review_8 = clean_data[REVIEW_COLUMN_NAME].iloc[7]

    # Process the text using spaCy
    doc_5 = nlp(review_5)
    doc_8 = nlp(review_8)

    # Calculate the similarity between the two reviews
    similarity_score = doc_5.similarity(doc_8)

    # Print the similarity score
    print(f"Similarity between row 5 and row 8: {similarity_score}")
