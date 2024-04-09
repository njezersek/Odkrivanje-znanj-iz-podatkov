from os import listdir
from os.path import join

from unidecode import unidecode

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import sklearn.manifold

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
#     'savefig.bbox': 'tight',
#     'savefig.pad_inches': 0.0,
# })

def terke(text, n):
    """
    Vrne slovar s preštetimi terkami dolžine n.
    """
    text = unidecode(text)
    text = text.lower()
    # count the frequencies of substrings length n in a string
    freq = {}
    for i in range(len(text) - n + 1):
        sub = text[i:i+n]
        if sub in freq:
            freq[sub] += 1
        else:
            freq[sub] = 1
    return freq


def read_data(n_terke):
    # Prosim, ne spreminjajte te funkcije. Vso potrebno obdelavo naredite
    # v funkciji terke.
    lds = {}
    for fn in listdir("jeziki"):
        if fn.lower().endswith(".txt"):
            with open(join("jeziki", fn), encoding="utf8") as f:
                text = f.read()
                nter = terke(text, n=n_terke)
                lds[fn] = nter
    return lds


# cosine distance between two dictionary of frequencies
def cosine_dist(d1, d2):
    intersection = set(d1) & set(d2)

    # if there is no intersection, return 1
    if len(intersection) == 0:
        return 1
    
    v1 = np.array([d1[k] for k in intersection])
    v2 = np.array([d2[k] for k in intersection])

    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def idf(term, data_dict):
    """
    Compute inverse document frequency of a term.
    """
    N = len(data_dict)
    n = 0
    for lang, freq in data_dict.items():
        if term in freq:
            n += 1
    return np.log(N / n)


def prepare_data_matrix(data_dict):
    """
    Return data in a matrix (2D numpy array), where each row contains triplets
    for a language. Columns should be the 100 most common triplets
    according to the idf (NOT the complete tf-idf) measure.
    """
    # find the 100 most common triplets according to the idf measure
    all_triplets = set()
    for _, freq in data_dict.items():
        for triplet in freq:
            all_triplets.add(triplet)

    all_triplets_with_idf = [(triplet, idf(triplet, data_dict)) for triplet in all_triplets]

    most_common_tripets = sorted(all_triplets_with_idf, key=lambda x: x[1])[:100]

    X = np.zeros((len(data_dict), len(most_common_tripets)))
    languages = list(data_dict.keys())
    for i, lang in enumerate(languages):
        for j, (triplet, _) in enumerate(most_common_tripets):
            X[i, j] = data_dict[lang].get(triplet, 0)

    return X, languages


def power_iteration(X):
    """
    Compute the eigenvector with the greatest eigenvalue
    of the covariance matrix of X (a numpy array).

    Return two values:
    - the eigenvector (1D numpy array) and
    - the corresponding eigenvalue (a float)
    """

    cov = np.cov(X, rowvar=False)
    u = np.ones(X.shape[1])
    for _ in range(100):
        u_new = cov @ u
        if np.linalg.norm(u_new - u) < 1e-12:
            break
        
        u = u_new / np.linalg.norm(u_new)

    return u, np.dot(u, cov @ u)


def power_iteration_two_components(X):
    """
    Compute first two eigenvectors and eigenvalues with the power iteration method.
    This function should use the power_iteration function internally.

    Return two values:
    - the two eigenvectors (2D numpy array, each eigenvector in a row) and
    - the corresponding eigenvalues (a 1D numpy array)
    """
    vec1, val1 = power_iteration(X)

    v = np.array([vec1])

    X2 = X - (X @ v.T @ v)
    vec2, val2 = power_iteration(X2)

    vec = np.stack((vec1, vec2))
    val = np.array([val1, val2])

    return vec, val

def project_to_eigenvectors(X, vecs):
    """
    Project matrix X onto the space defined by eigenvectors.
    The output array should have as many rows as X and as many columns as there
    are vectors.
    """
    mean = np.mean(X, axis=0, keepdims=True)
    return (X - mean) @ vecs.T


def total_variance(X):
    """
    Total variance of the data matrix X. You will need to use for
    to compute the explained variance ratio.
    """
    return np.var(X, axis=0, ddof=1).sum()


def explained_variance_ratio(X, eigenvectors, eigenvalues):
    """
    Compute explained variance ratio.
    """
    return np.sum(eigenvalues) / total_variance(X)


def plot(proj, languages, title="", filename=""):
    langage_codes = [l.split("-")[0] for l in languages]

    color_lang = {
        "C0": ["en", "de", "da", "sv", "nl"], # germanski
        "C1": ["fr", "es", "it", "ca", "pt",], # romanski
        "C2": ["uk", "sr", "sl", "cs", "ru", "pl"], # slovanski
    }

    # plot projection
    # scatter label
    plt.figure()
    for i, lang in enumerate(langage_codes):
        color = "C4"
        for k, v in color_lang.items():
            if lang in v:
                color = k

        plt.scatter([proj[i, 0]], [proj[i, 1]], c=[color])
        plt.annotate(langage_codes[i], (proj[i, 0], proj[i, 1]))
    
    plt.title(title)
    plt.show()
    # if filename:
    #     plt.savefig(filename)

def plot_PCA():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of PCA on languages data.
    """
    data = read_data(n_terke=3)
    X, languages = prepare_data_matrix(data)

    # normalize X by rows
    X = X / X.sum(axis=1, keepdims=True)

    vec, val = power_iteration_two_components(X)

    proj = project_to_eigenvectors(X, vec)

    plot(proj, languages, title="PCA", filename="pca.pgf")


def plot_MDS():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of MDS on languages data.

    Use sklearn.manifold.MDS and explicitly run it with a distance
    matrix obtained with cosine distance on full triplets.
    """
    data = read_data(n_terke=3)
    
    # compute distance between all data points
    X = np.zeros((len(data), len(data)))
    for i, lang in enumerate(data):
        for j, lang2 in enumerate(data):
            X[i, j] = cosine_dist(data[lang], data[lang2])

    languages = list(data.keys())

    # normalize X by rows
    X = X / X.sum(axis=1, keepdims=True)

    embeding = sklearn.manifold.MDS(n_components=2)
    proj = embeding.fit_transform(X)

    plot(proj, languages, title="MDS", filename="mds.pgf")


def plot_tSNE():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of tSNE on languages data.

    Use sklearn.manifold.TSNE and explicitly run it with a distance
    matrix obtained with cosine distance on full triplets.
    """
    data = read_data(n_terke=3)
    
    # compute distance between all data points
    X = np.zeros((len(data), len(data)))
    for i, lang in enumerate(data):
        for j, lang2 in enumerate(data):
            X[i, j] = cosine_dist(data[lang], data[lang2])

    languages = list(data.keys())

    # normalize X by rows
    X = X / X.sum(axis=1, keepdims=True)

    embeding = sklearn.manifold.TSNE(n_components=2)
    proj = embeding.fit_transform(X)

    plot(proj, languages, title="t-SNE", filename="tsne.pgf")

if __name__ == "__main__":
   plot_tSNE()
   plot_MDS()
   plot_PCA()