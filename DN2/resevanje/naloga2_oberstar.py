from os import listdir
from os.path import join
from unidecode import unidecode
import re

import numpy as np
import matplotlib.pyplot as plt


def terke(text, n):
	"""
	Vrne slovar s preštetimi terkami dolžine n.
	"""

	text = unidecode(text).lower()
	dictionary = {}

	for i in range(len(text) - n + 1):
		key = text[i:i+n]
		dictionary[key] = dictionary.get(key, 0) + 1
	
	return dictionary


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


def cosine_dist(d1, d2):
	"""
	Vrne kosinusno razdaljo med slovarjema terk d1 in d2.
	"""

	d1s = set(d1)
	d2s = set(d2)
	common = d1s.intersection(d2s)

	if len(common) == 0:
		return 1

	a1 = np.array([d1[i] for i in common])
	a2 = np.array([d2[i] for i in common])

	return 1 - a1.dot(a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))


def prepare_data_matrix(data_dict):
	"""
	Return data in a matrix (2D numpy array), where each row contains triplets
	for a language. Columns should be the 100 most common triplets
	according to the idf (NOT the complete tf-idf) measure.
	"""

	X = np.zeros((len(data_dict), 100))
	languages = data_dict.keys()

	global_dict = {}
	# flatten dict to single n-gram dictionary
	for language, ngram_dict in data_dict.items():
		for ngram, count in ngram_dict.items():
			global_dict[ngram] = global_dict.get(ngram, 0) + count

	top_ngrams = [i[0] for i in sorted(global_dict.items(),
									   key=lambda x: x[1], reverse=True)[:100]]
	global_dict = None

	for i, (language, ngram_dict) in enumerate(data_dict.items()):
		X[i] = np.array([ngram_dict.get(x, 0) for x in top_ngrams])

	# normalization factor
	s = np.array([np.sum(X, axis=1)])
	return (X / s.T) , languages


def power_iteration(X):
	"""
	Compute the eigenvector with the greatest eigenvalue
	of the covariance matrix of X (a numpy array).

	Return two values:
	- the eigenvector (1D numpy array) and
	- the corresponding eigenvalue (a float)
	"""

	cov_mat = np.cov(X.T)

	eigenvector = np.random.rand(cov_mat.shape[0])
	prev = np.zeros(cov_mat.shape[0])

	while(np.linalg.norm(eigenvector - prev) > 1e-4):
		prev = eigenvector

		new_eigenvector = np.dot(cov_mat, eigenvector)
		eigen_norm = np.linalg.norm(new_eigenvector)
		eigenvector = new_eigenvector / eigen_norm

	return eigenvector, (np.dot(eigenvector, cov_mat @ eigenvector)) / np.dot(eigenvector, eigenvector)


def power_iteration_two_components(X):
	"""
	Compute first two eigenvectors and eigenvalues with the power iteration method.
	This function should use the power_iteration function internally.

	Return two values:
	- the two eigenvectors (2D numpy array, each eigenvector in a row) and
	- the corresponding eigenvalues (a 1D numpy array)
	"""
	vec_a, lambda_a = power_iteration(X)
	vec_a = np.array([vec_a])
	vec_b, lambda_b = power_iteration(X - (X @ vec_a.T @ vec_a))

	return np.vstack([vec_a, vec_b]), np.array([lambda_a, lambda_b])


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


def plot_PCA():
	"""
	Everything (opening files, computation, plotting) needed
	to produce a plot of PCA on languages data.
	"""
	data_dict = read_data(3)
	
	X, languages = prepare_data_matrix(data_dict)
	vecs, vals = power_iteration_two_components(X)
	proj = project_to_eigenvectors(X, vecs)

	# obtaining language codes and colors
	lang_codes = [l.split('_')[0] for l in languages]
	lang_colors = {
		'C0' : ['cs', 'pl', 'ru', 'sl', 'sr', 'uk'], 	# slovanski
		'C1' : ['da', 'de', 'en', 'nl', 'sv'], 			# germanski
		'C2' : ['ca', 'es', 'fr', 'it', 'pt'], 			# romanski
		'C3' : ['ar', 'hi', 'hu', 'si', 'zh'] 			# other
	}

	for i, language in enumerate(languages):
		color = 'C4'
		lang_split = language.split('_')[0]

		for k, v in lang_colors.items():
			if lang_split in v:
				color = k
				break

		plt.scatter(proj[i, 0], proj[i, 1], c=[color])
		plt.annotate(lang_codes[i], (proj[i, 0], proj[i, 1]))
		
	plt.show()


def plot_MDS():
	"""
	Everything (opening files, computation, plotting) needed
	to produce a plot of MDS on languages data.

	Use sklearn.manifold.MDS and explicitly run it with a distance
	matrix obtained with cosine distance on full triplets.
	"""
	pass
	# ...


if __name__ == "__main__":
#    plot_MDS()
	plot_PCA()