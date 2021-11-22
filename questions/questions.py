import nltk
import sys

from nltk.tokenize import word_tokenize

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    import os
    files_dict = {}
    for file in os.listdir(directory):
        # Consider only text files
        if file.endswith(".txt"):
            with open(os.path.join(directory, file)) as f:
                files_dict[file] = f.read()

    return files_dict

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    import string
    punct = string.punctuation
    stopwords = nltk.corpus.stopwords.words("english")
    tokenized = word_tokenize(document.lower())
    tokenized_final = []
    for token in tokenized:
        # If token is neither a punctuation nor a stopword
        if token not in punct and token not in stopwords:
            tokenized_final.append(token)
    return tokenized_final

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    import math
    word_idf = {}
    num_docs = len(documents)
    for text in documents.values():
        for word in text:
            # Call to helper function count_docs
            num_docs_with_word = count_docs(documents, word)
            idf = math.log(num_docs / num_docs_with_word)
            word_idf[word] = idf
    return word_idf

def count_docs(documents, word):
    """ 
    Helper function to count the number of documents in which the given word appears
    given a dictionary 'documents' that maps names of documents to a list of words.
    """
    count = 0
    for words_list in documents.values():
        if word in words_list:
            count += 1
    return count

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf = {filename:0 for filename in files.keys()}
    for filename, words in files.items():
        for word in query:
            if word in words:
                # Calculating tf-idf
                tf = words.count(word)
                idf = idfs[word]
                tf_idf[filename] += (tf * idf)

    # Sorting tf_idf dictionary in the descending order of tf-idf values
    tf_idf_sorted = dict(sorted(tf_idf.items(), key=lambda x: x[1], reverse=True))

    # Creating list of filenames of top n files ranked according to their tf-idf values
    tf_idf_top_n = list(tf_idf_sorted.keys())[:n]

    return tf_idf_top_n
            
def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_measures = {}
    for sentence, word_list in sentences.items():
        matching_word_measure = 0
        query_term_density = 0
        query_words = 0
        num_words = len(word_list)
        for word in query:
            if word in word_list:
                query_words += 1
                matching_word_measure += idfs[word]
        
        query_term_density = query_words / num_words

        # Dictionary of sentences mapped to a list of their matching word measure and query term density
        sentence_measures[sentence] = [matching_word_measure, query_term_density]

    # Sorting will happen in order of elements in the list (key of the dictionary)
    # That is, sort first by matching word measure, if tie occurs then sort by query term density
    sorted_sentence_measures = dict(sorted(sentence_measures.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True))

    # Creating list of top n sentences from the sorted dictionary
    sentences_top_n = list(sorted_sentence_measures.keys())[:n]

    return sentences_top_n

if __name__ == "__main__":
    main()
