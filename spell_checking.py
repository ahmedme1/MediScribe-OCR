import nltk
from nltk.corpus import words

# Download the 'words' corpus if not already downloaded
nltk.download('words')

# Load the list of words from the 'words' corpus
word_list = set(words.words())

def correct_text(text):
    corrected = " ".join([word if word in word_list else "[Unrecognized]" for word in text.split()])
    return corrected

# Example usage
if __name__ == "__main__":
    sample_text = "This is a smple text with sme speling erors."
    corrected_text = correct_text(sample_text)
    print("Original Text:", sample_text)
    print("Corrected Text:", corrected_text)
