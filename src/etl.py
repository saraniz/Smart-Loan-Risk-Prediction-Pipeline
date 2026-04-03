import pandas as pd
from pathlib import Path
import re

DATA_PATH='../data'
CLEANED_DATA_PATH='../data/cleaned_loan.csv'

#load data
def load_data():
    df = pd.read_csv('../data/financial_loan.csv')
    print(df.head)
    print(df.columns)

    return df

def clean_data(df):
    df = df.copy()

    print("Null values per column", df.isnull().sum())

    # select most important featurecolumns
    df = df[[
        'annual_income',
         'loan_amount',
         'dti',
         'emp_length',
         'home_ownership',
         'purpose',
         'int_rate',
         'installment',
         'loan_status',
         'term'
    ]]

    df.fillna(df.mean(numeric_only=True), inplace=True)

    # select_dtypes(include=['object']) finds all columns that contain Text/Strings
    categorical_cols = df.select_dtypes(include=['object']).columns
    # Fills missing text values with a placeholder so the model doesn't error out
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')

    print(df.columns)
    
    return df

def feature_engineering(df):
    if df is None: return None
    df = df.copy()

    def clean_emp_len(x):
        # Handle nulls or placeholders first
        if pd.isna(x) or x == 'Unknown': return 0
        
        # Convert to lowercase to make searching easier (case-insensitive)
        x = str(x).lower()
        
        # Logic: If they have less than a year, we count it as 0 years of stability
        if '<' in x: return 0
        # Logic: If it says '10+', we treat it as 10 (the max integer)
        if '+' in x: return 10
        
        # re.findall(r'\d+', x) looks for digits (\d) in the string
        # r'\d+' means "find one or more numbers in a row"
        nums = re.findall(r'\d+', x)
        
        # If a number was found, return it as an integer, else return 0
        return int(nums[0]) if nums else 0

    # .apply() runs the 'clean_emp_len' function on every single row in that column
    df['emp_length'] = df['emp_length'].apply(clean_emp_len)

    # Formula: (Monthly Payment * 12) / Annual Income
    # We add 1 to income to avoid 'ZeroDivisionError' if a row has 0 income
    df['installment_income_ratio'] = (df['installment'] * 12) / (df['annual_income'] + 1)

    # We only want to train on finished loans. 'Current' loans tell us nothing yet.
    # df[df['col'] != 'val'] filters the rows to REMOVE 'Current' status
    df = df[df['loan_status'] != 'Current']
    
    # Dictionary mapping: Converts English status labels into Machine-Readable bits (0 and 1)
    status_map = {
        'Fully Paid': 0,      # Good user
        'Charged Off': 1,     # Defaulted (Bad user)
        'Default': 1,         # Defaulted
        'Late (31-120 days)': 1, 
        'Late (16-30 days)': 1,
        'In Grace Period': 0  # Still technically a 'Good' status for most banks
    }
    
    # .map() replaces the text in the column with the numbers from our dictionary
    df['target'] = df['loan_status'].map(status_map)
    
    # .drop() removes the old text column since we now have the numeric 'target' column
    # axis=1 or columns=['name'] tells pandas to drop a vertical column, not a horizontal row
    df = df.drop(columns=['loan_status'])

    return df

def save_data(df):
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Cleaned data saved to {CLEANED_DATA_PATH}")

if __name__ == "__main__":
    # Load raw data
    df = load_data()
    
    # Clean the data
    df = clean_data(df)
    
    # Engineer features
    df = feature_engineering(df)
    
    # Save cleaned data
    save_data(df)
    
    print("ETL process completed successfully!")

if __name__ == "__main__":

    print("ETL Pipeline loading...")
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)

    print("ETL Pipeline successfully loaded...")