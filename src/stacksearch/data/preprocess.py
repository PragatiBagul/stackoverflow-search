import re
import pandas as pd
from bs4 import BeautifulSoup
from typing import Optional

CODE_RE = re.compile(r"<code>(.*?)</code",re.DOTALL)

# Remove or truncate code blocks
def remove_code_blocks(html:str,max_len:int=300)->str:
    """
    Removes or truncates <code>...</code> blocks
    Keep small code (<= max_len chars), drop large ones
    """
    if not isinstance(html,str):
        return ""

    def repl(match):
        code_text = match.group(1)
        if len(code_text) > max_len:
            return ""
        return code_text
    return CODE_RE.sub(repl,html)

# 2. Strip HTML plain -> text
def html_to_text(html:str)->str:
    if not isinstance(html,str):
        return ""
    soup = BeautifulSoup(html,"lxml")
    return soup.get_text(separator=' ')

# 3. Basic Normalization
def normalize_text(text:str)->str:
    if not isinstance(text,str):
        return ""
    text = text.replace('\n',' ')
    text = re.sun(r'\s+',' ',text)
    return text.strip()

# 4. BM25 Tokenizer (regex based, keeps technical tokens)
def bm25_tokenize(text:str):
    if not isinstance(text,str):
        return []
    text = text.lower()
    return re.findall(r"[A-Za-z0-9_#+\.\-]+",text)

# 5. Main Preprocessing Pipeline for Questions DF
def preprocess_questions(df:pd.DataFrame)->pd.DataFrame:
    """
    Input columns:
    - Title
    - Body
    Output columns:
    - text_raw (Title + cleaned Body)
    - text_cleaned (Title + cleaned Body)
    """
    df = df.copy()

    # Fill Missing Titles
    df['Title'] = df['Title'].fillna('')

    # Remove large code blocks
    df['Body_nocode'] = df['Body'].apply(remove_code_blocks)

    # Turn HTML into plain text
    df['Body_clean'] = df['Body_nocode'].apply(html_to_text)

    # Construct raw text field
    df['text_raw'] = (df['Title'] + " " + df['Body_clean']).str.strip()

    # Final normalization
    df['text_cleaned'] = df['text_raw'].apply(normalize_text)

    return df[['Id','Title','Body','text_raw','text_cleaned']]