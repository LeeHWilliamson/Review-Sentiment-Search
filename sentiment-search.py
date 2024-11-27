import pandas as pd

# Read the DataFrame from the pickle file
df = pd.read_pickle("reviews_segment.pkl")

# Output the DataFrame first 5 columns
df.head()