import pandas as pd
import re
import os
import glob

def extract_answer_robust(text):
    """
    Function to robustly extract answers like "The answer is B", "Option A", etc.
    """
    if not isinstance(text, str):
        return None
    
    text = text.strip()
    
    # Priority 1: "answer is [X]" or "answer: [X]" (handles optional LaTeX like $\text{X}$)
    pattern_answer = r"answer\s+(?:is)?\s*:?\s*(?:[\$\(]*\\?text\{?)?\s*([A-E])\b"
    
    # Priority 2: "Option [X]"
    pattern_option = r"option\s+([A-E])\b"
    
    # Priority 3: Just the letter if it's the very first character or the whole string matches
    pattern_simple = r"^([A-E])[\.\)]?$"

    match = re.search(pattern_answer, text, re.IGNORECASE)
    if match: return match.group(1).upper()
        
    match = re.search(pattern_option, text, re.IGNORECASE)
    if match: return match.group(1).upper()

    match = re.search(pattern_simple, text, re.IGNORECASE)
    if match: return match.group(1).upper()
            
    return None

def process_csv_files(input_dir='input', output_dir='output'):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all CSV files
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in directory: {input_dir}")
        return

    print(f"Found {len(csv_files)} files. Processing...")

    for file_path in csv_files:
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            file_name = os.path.basename(file_path)
            
            # Capture original column order to preserve it exactly
            original_columns = df.columns.tolist()

            # Ensure 'target' column exists and is clean
            if 'target' in df.columns:
                df['target'] = df['target'].astype(str).str.strip()
            else:
                print(f"Skipping {file_name}: 'target' column missing.")
                continue

            # Identify agents dynamically by finding '_response' columns
            response_columns = [col for col in df.columns if col.endswith('_response')]
            experts = [col.replace('_response', '') for col in response_columns]
            
            # List to track new columns we create (to append them at the end)
            new_columns_ordered = []

            for expert in experts:
                response_col = f'{expert}_response'
                
                # Determine the score column name.
                # If a score column already exists (e.g. 'Malicious_score'), we update it in place.
                # If not, we create '{expert}_new_score'.
                score_col = f'{expert}_new_score'

                extracted_col = f'{expert}_extracted_answer'
                
                # 1. Extract Answer
                df[extracted_col] = df[response_col].apply(extract_answer_robust)
                
                # 2. Calculate Score (Matches target = 1, else 0)
                # This updates the existing column if it exists, or creates a new one
                df[score_col] = (df[extracted_col] == df['target']).astype(int)

                # 3. Apply Malicious Agent Penalty
                # If agent is Malicious and called tools > 0 times, score becomes 0
                if expert.lower() == 'malicious' and 'malicious_calls' in df.columns:
                    # Ensure numeric
                    df['malicious_calls'] = pd.to_numeric(df['malicious_calls'], errors='coerce').fillna(0)
                    
                    mask_penalty = df['malicious_calls'] > 0
                    if mask_penalty.any():
                        df.loc[mask_penalty, score_col] = 0
                        print(f"  - Applied penalty to 'Malicious' in {file_name} (rows: {mask_penalty.sum()})")

                # Track new columns for final ordering
                if extracted_col not in original_columns and extracted_col not in new_columns_ordered:
                    new_columns_ordered.append(extracted_col)
                if score_col not in original_columns and score_col not in new_columns_ordered:
                    new_columns_ordered.append(score_col)

            # Reconstruct DataFrame with exact original order + new columns appended
            final_columns = original_columns + new_columns_ordered
            df = df[final_columns]

            # Save
            output_path = os.path.join(output_dir, file_name)
            df.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    process_csv_files()
