import nltk
from nltk.tokenize import sent_tokenize

# Make sure to run this once if you haven't: nltk.download('punkt')

def chunk_text(text, max_chunk_size=500):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = ""
    
    for sentence in sentences:
        if len(chunk) + len(sentence) > max_chunk_size:
            chunks.append(chunk.strip())
            chunk = sentence
        else:
            chunk += " " + sentence

    if chunk:
        chunks.append(chunk.strip())
    
    return chunks