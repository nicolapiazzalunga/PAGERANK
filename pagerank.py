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
    transition_distribution = dict()
    d = damping_factor
    m = len(corpus[page])
    N = len(corpus)
    for link in corpus:
        if link in corpus[page]:
            transition_distribution[link] = d / m + (1 - d) / N
        else:
            transition_distribution[link] = (1 - d) / N
    return transition_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    sample = dict()

    # Initialize the transition distribution with a randomly chosen page
    page = random.choice(list(corpus.items()))[0]
    transition_distribution = transition_model(corpus, page, damping_factor)

    # Begin sampling n times
    iteration = 0
    while iteration < n:

        # Extract a sample page
        population = list()
        probabilities = list()
        for link, probability in transition_distribution.items():
            population.append(link)
            probabilities.append(probability)
        page = random.choices(population, probabilities)[0]

        # Update sample PageRank
        if page in sample:
            sample[page] = sample[page] + 1 / n
        else:
            sample[page] = 1 / n

        # Update transition_distribution given new sample page
        transition_distribution = transition_model(corpus, page, damping_factor)

        # Increment counter
        iteration += 1
    
    # Return sample PageRank
    return sample


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    current_page_rank = dict()
    new_page_rank = dict()

    # Set up
    N = len(corpus)
    d = damping_factor
    const_prob = (1 - d) / N

    # Initialize prev_page_rank with equal probability
    for page in corpus:
        current_page_rank[page] = 1 / N
    
    # Initialize next_page_rank with one cycle of the iteration
    for p in corpus:
        summation = 0
        for i in corpus:
            if len(corpus[i]) == 0:
                summation += current_page_rank[i] / N    
            elif p in corpus[i]:
                num_links = len(corpus[i])
                summation += current_page_rank[i] / num_links
            else:
                continue     
        new_page_rank[p] = const_prob + d * summation

    # Main iteration    
    while not all([abs(new_page_rank[page] - current_page_rank[page]) <= 0.001 for page in corpus]):
        current_page_rank = new_page_rank.copy()
        for p in corpus:
            summation = 0
            for i in corpus:
                if len(corpus[i]) == 0:
                    summation += current_page_rank[i] / N
                elif p in corpus[i]:
                    num_links = len(corpus[i])
                    summation += current_page_rank[i] / num_links      
                else:
                    continue 
            new_page_rank[p] = const_prob + d * summation
    
    return current_page_rank


if __name__ == "__main__":
    main()
