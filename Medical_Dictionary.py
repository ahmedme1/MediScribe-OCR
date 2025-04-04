import nltk

# Example list of medical terms (you can expand this list)
medical_terms_list = [
    "diabetes mellitus", "hypertension", "asthma", "cancer", "heart attack",
    "stroke", "arthritis", "depression", "anxiety", "obesity"
]

# Convert the list to a set for faster lookup
medical_terms_set = set(medical_terms_list)

def medical_term_check(text):
    words = nltk.word_tokenize(text)
    recognized_terms = [word for word in words if word.lower() in medical_terms_set]
    return recognized_terms

# Example usage
if __name__ == "__main__":
    nltk.download('punkt')  # Download the tokenizer data
    sample_text = "The patient was diagnosed with diabetes mellitus and hypertension."
    medical_terms = medical_term_check(sample_text)
    print("Medical Terms:", medical_terms)
