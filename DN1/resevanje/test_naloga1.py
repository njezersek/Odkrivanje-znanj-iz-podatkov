import os
import shutil
import time
import tempfile
import unittest

import numpy as np
from PIL import Image

t = time.time()
import naloga1
if time.time() - t > 20:
    raise Exception("Uvoz vaše kode traja več kot dve sekundi. "
                    "Pazite, da se ob uvozu koda ne požene.")


from naloga1 import read_data, cosine_dist, k_medoids, silhouette,\
    silhouette_average



class Naloga1Test(unittest.TestCase):

    def test_read_data(self):
        # Preveri samo format, ne pa vsebine
        tempdir = tempfile.mkdtemp()
        try:
            data = np.zeros((10, 10, 3), dtype=np.uint8)
            data[0:3, 0:3] = [255, 0, 0]
            img = Image.fromarray(data, 'RGB')
            img.save(os.path.join(tempdir, 'image1.png'))
            img.save(os.path.join(tempdir, 'image2.jpg'))
            data = read_data(tempdir)
            self.assertIsInstance(data, dict)
            self.assertEqual(len(data), 2)
            self.assertIn('image1.png', data)
            self.assertIn('image2.jpg', data)
            for v in data.values():
                # either a 1D numpy array os a pytorch tensor
                shape = list(v.shape)
                self.assertEqual(shape, [1000])
        finally:
            # remove the temp images
            shutil.rmtree(tempdir)

    def test_cosine_dist(self):
        d1 = np.array([1, 1, 0, 0])
        d2 = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(cosine_dist(d1, d2), 1)
        d1 = np.array([1, 1, 0, 0])
        d2 = np.array([1, 1, 0, 0])
        self.assertAlmostEqual(cosine_dist(d1, d2), 0)

    def test_k_medoids(self):
        data = {"X": np.array([1, 1]),
                "Y": np.array([0.9, 1]),
                "Z": np.array([1, 0])}

        clusters = k_medoids(data, ["X", "Z"])  # dva medoida
        self.assertEqual(2, len(clusters))  # dva clustra na koncu
        self.assertIn(["Z"], clusters)  # poseben element bo ločen cluster
        self.assertTrue(["X", "Y"] in clusters or ["Y", "X"] in clusters)

        clusters = k_medoids(data, ["X", "Y", "Z"])  # trije medoidi
        self.assertEqual(3, len(clusters))  # trije clustri na koncu
        self.assertIn(["X"], clusters)
        self.assertIn(["Y"], clusters)
        self.assertIn(["Z"], clusters)

    def test_silhouette(self):
        data = {"X": np.array([1, 1]),
                "Y": np.array([0.9, 1]),
                "Z": np.array([1, 0])}

        s1 = silhouette("X", [["X", "Y"], ["Z"]], data)
        self.assertTrue(0.5 < s1 < 1)

    def test_silhouette_average(self):
        data = {"X": np.array([1, 1]),
                "Y": np.array([0.9, 1]),
                "Z": np.array([1, 0])}

        s1 = silhouette_average(data, [["X", "Y"], ["Z"]])  # boljše skupine
        s2 = silhouette_average(data, [["X", "Z"], ["Y"]])  # slabše skupine
        s3 = silhouette_average(data, [["Y", "Z"], ["X"]])  # še slabše skupine
        self.assertLess(s2, s1)
        self.assertLess(s3, s2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
