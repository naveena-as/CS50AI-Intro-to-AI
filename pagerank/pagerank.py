import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # dictionary to store the distribution
    dist = {}
    # total number of pages in the corpus
    pages = len(corpus)
    random_probability = (1-damping_factor) / pages
    # number of links from the page
    links = len(corpus[page])
    if links != 0:
        #link_probability = ((1-random_probability)/links)
        #link_probability = damping_factor / len(corpus[page])
        for _page in corpus:
            dist[_page] = random_probability
        for link in corpus[page]:
            dist[link] += damping_factor / links
    else:
        for _page in corpus:
            dist[_page] = 1/pages
    return dist

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # dictionary to store sample pagerank values
    sample_pr = {}
    for page in corpus:
        sample_pr[page] = 0 
    # first sample chosen at random
    sample = random.choice(list(corpus))
    for i in range(1,n):
        dist = transition_model(corpus, sample, damping_factor)
        for sample in sample_pr:
            sample_pr[sample] = (((i-1) * sample_pr[sample]) + dist[sample]) / i
        next_page = list(sample_pr.keys())
        probability = list(sample_pr.values())
        sample = random.choices(next_page, weights = probability)[0]
    # normalising
    total = sum(sample_pr.values())
    for page in sample_pr.keys():
        sample_pr[page] = sample_pr[page]/total
    return sample_pr


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = len(corpus)
    iterate_pr = {}
    for page in corpus:
        iterate_pr[page] = (1/pages)
    # boolean variable to check if difference between current and new rpagerank values is > 0.001
    diff = True
    #iterate till difference becomes < 0.001
    while diff:
        old_pr = dict(iterate_pr)
        for page in iterate_pr:
            # creating a list of pages linked to a page
            linked_pages = []
            for _page in corpus:
                if page in corpus[_page]:
                    linked_pages.append(_page)
            calculated_sum = 0
            for _page in linked_pages:
                calculated_sum += iterate_pr[_page] / len(corpus[_page])
            iterate_pr[page] =  ((1-damping_factor)/pages) + (damping_factor * calculated_sum)
            diff = bool(abs(old_pr[page]-iterate_pr[page]) > 0.001)
    # normalising
    total = sum(iterate_pr.values())
    for page in iterate_pr.keys():
        iterate_pr[page] = iterate_pr[page]/total
    return iterate_pr

if __name__ == "__main__":
    main()
