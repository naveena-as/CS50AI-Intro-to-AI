import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> S Conj S | NP VP | NP VP PP | NP VP NP | S Conj VP NP | VP PP | NP VP PP Adv
NP -> N | Det N | NP PP | Det AdjP N
VP -> V | V Adv | Adv V | VP NP
PP -> P NP
AdjP -> Adj | Adj AdjP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # List of tokenized words
    tokenized = nltk.word_tokenize(sentence)
    # List to store the elements to be removed: that does not contain atleast one alphabetic character
    to_remove = []
    for token in tokenized:
        # If element has only digits or is not a combination of numbers and alphabets / alphabets alone
        if token.isdigit() or (token.isalnum() == False):
            to_remove.append(token)
    # New list after removing unnecessary elements in lower case
    tokenized_new = [x.lower() for x in tokenized if x not in to_remove]
    return tokenized_new



def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    np_chunks = []
    # Finding subtrees with label as "NP"
    for np_subtree in tree.subtrees(lambda t: t.label() == "NP"):
        valid_append = True
        # Checking further subtrees to see if it itself contains other NP subtree
        for subtree in np_subtree.subtrees():
            # If label of subtree is NP and it is not the same tree itself, do not append that tree
            if subtree.label() == "NP" and subtree != np_subtree:
                valid_append = False
                break
        # If it is a valid NP subtree, add to the list of NP chunks
        if valid_append:
            np_chunks.append(np_subtree)
    return np_chunks


if __name__ == "__main__":
    main()
