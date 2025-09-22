
import pandas as pd
import numpy as np

def clean_titanic_dataset(input_file, output_file):
    """
    Complete data cleaning pipeline for Titanic dataset
    """
    print("=== TITANIC DATA CLEANING PIPELINE ===")

    # Load dataset
    df = pd.read_csv(input_file)
    print(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Create working copy
    df_clean = df.copy()

    # Step 1: Handle missing values
    print("\n1. Handling missing values...")
    df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
    df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)
    df_clean['Has_Cabin'] = df_clean['Cabin'].notna().astype(int)

    # Step 2: Drop unnecessary columns
    print("2. Removing unnecessary columns...")
    df_clean.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # Step 3: Standardize text values
    print("3. Standardizing text values...")
    df_clean['Sex'] = df_clean['Sex'].str.lower()
    df_clean['Embarked'] = df_clean['Embarked'].str.upper()

    # Step 4: Rename columns
    print("4. Renaming columns...")
    column_mapping = {
        'Pclass': 'passenger_class',
        'Sex': 'gender', 
        'Age': 'age',
        'SibSp': 'siblings_spouses',
        'Parch': 'parents_children',
        'Fare': 'fare',
        'Embarked': 'port_embarked',
        'Survived': 'survived',
        'Has_Cabin': 'has_cabin'
    }
    df_clean.rename(columns=column_mapping, inplace=True)

    # Step 5: Optimize data types
    print("5. Optimizing data types...")
    df_clean['passenger_class'] = df_clean['passenger_class'].astype('int8')
    df_clean['age'] = df_clean['age'].astype('int8')
    df_clean['siblings_spouses'] = df_clean['siblings_spouses'].astype('int8')
    df_clean['parents_children'] = df_clean['parents_children'].astype('int8')
    df_clean['survived'] = df_clean['survived'].astype('int8')
    df_clean['has_cabin'] = df_clean['has_cabin'].astype('int8')

    # Step 6: Remove duplicates
    print("6. Removing duplicate rows...")
    original_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates(keep='first')
    duplicates_removed = original_rows - len(df_clean)

    # Save cleaned dataset
    df_clean.to_csv(output_file, index=False)

    print(f"\n=== CLEANING COMPLETE ===")
    print(f"Final dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Missing values: {df_clean.isnull().sum().sum()}")
    print(f"Cleaned dataset saved as: {output_file}")

    return df_clean

# Usage example
if __name__ == "__main__":
    cleaned_df = clean_titanic_dataset('Titanic-Dataset.csv', 'titanic_cleaned.csv')
    print("\nCleaning pipeline executed successfully!")
