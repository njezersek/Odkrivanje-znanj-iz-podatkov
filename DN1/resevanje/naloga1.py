import sys
import torch
from torchvision import transforms
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import random
import math

def read_data(path):
    """
    Preberi vse slike v podani poti, jih pretvori v "embedding" in vrni rezultat
    kot slovar, kjer so ključi imena slik, vrednosti pa pripadajoči vektorji.
    """
    images_dir = Path(path)

    squeeznet = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)

    img_dict = {}

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for img_path in images_dir.iterdir():
        img = Image.open(str(img_path))
        img_preprocessed = preprocess(img)
        input_batch = img_preprocessed.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            squeeznet.to('cuda')
        
        with torch.no_grad():
            output = squeeznet(input_batch)
            img_dict[img_path.name] = output[0].cpu().detach().numpy()

    return img_dict


def cosine_dist(a, b):
    """
    Vrni razdaljo med vektorjema a in b, ki je smiselno
    pridobljena iz kosinusne podobnosti.
    """
    return 1-a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b))


### implementacija k_medoids iz wikipedie: https://en.wikipedia.org/wiki/K-medoids
### Partitioning Around Medoids (PAM)


def get_clusters(data, medoids):
    clusters = {}

    for medoid in medoids:
        clusters[medoid] = []

    for key, val in data.items():
        min_d = np.inf
        nearest_medoid = ""
        for medoid in medoids:
            d = cosine_dist(val, data[medoid])
            if d < min_d:
                min_d = d
                nearest_medoid = medoid
        clusters[nearest_medoid].append(key)

    return clusters

def compute_cost(data, medoids):
    clusters = get_clusters(data, medoids)
    sum_d = 0
    for medoid, cluster in clusters.items():
        for point in cluster:
            sum_d += cosine_dist(data[point], data[medoid])
    return sum_d


def k_medoids_pam(data, medoids):
    """
    Za podane podatke (slovar vektorjev) in medoide vrni končne skupine
    kot seznam seznamov nizov (ključev v slovarju data).
    """
    cost = compute_cost(data, medoids)
    best_medoids = medoids.copy()

    changed = True
    while changed:
        changed = False
        for key, val in data.items():
            if key in medoids: continue
            for medoid in medoids:
                new_medoids = medoids.copy()
                new_medoids.remove(medoid)
                new_medoids.append(key)
                new_cost = compute_cost(data, new_medoids)

                if new_cost < cost:
                    changed = True
                    cost = new_cost
                    best_medoids = new_medoids

    return list(get_clusters(data, best_medoids).values())


### implementacija k medoids z Voronojevo iteracijo

def k_medoids(data, medoids):
    changed = True
    clusters = []
    while changed:
        changed = False
        # določi gruče
        clusters = [[] for _ in medoids]
        for p in data:
            best_i = -1
            best = np.inf
            for i, medoid in enumerate(medoids):
                d = cosine_dist(data[p], data[medoid])
                if d < best:
                    best = d
                    best_i = i
            clusters[best_i].append(p)


        # izračunaj nove medoide
        for i, cluster in enumerate(clusters):
            min_d = np.inf
            best_p = medoids[i]
            for p in cluster:
                sum_d = 0
                for q in cluster:
                    sum_d += cosine_dist(data[p], data[q])
                avg_d = sum_d/len(cluster)

                if avg_d < min_d:
                    avg_d = min_d
                    best_p = p
            
            if best_p != medoids[i]:
                changed = True
                medoids[i] = best_p
            
    return clusters

def silhouette(el, clusters, data):
    """
    Za element el ob podanih podatke (slovar vektorjev) in skupinah
    (seznam seznamov nizov: ključev v slovarju data), vrni silhueto za element el.
    """
    a = 0
    bs = []
    for cluster in clusters:
        if el in cluster:
            for n in cluster:
                a += cosine_dist(data[el], data[n])
            a /= len(cluster)
        else:
            b = 0
            for n in cluster:
                b += cosine_dist(data[el], data[n])
            bs.append(b / len(cluster))

    b = max(bs)

    return (b-a)/max(b,a)


def silhouette_average(data, clusters):
    """
    Za podane podatke (slovar vektorjev) in skupine (seznam seznamov nizov:
    ključev v slovarju data) vrni povprečno silhueto.
    """
    s = 0
    for el in data:
        s += silhouette(el, clusters, data)
    return s / len(data)


def visualize_clusters(clusters):
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.0,
    })
    for i, cluster in enumerate(clusters):
        cols = 7
        rows = math.ceil(len(cluster) / cols)
        fig = plt.figure(figsize=(20, 3*rows))
        fig.set_size_inches(w=10, h=1.5*rows)
        for j, img_path in enumerate(sorted(cluster, key=lambda n: silhouette(n, clusters, data), reverse=True)):
            plt.subplot(rows, cols, j+1)
            plt.axis('off')
            plt.imshow(Image.open(images_dir / img_path))
            plt.title(f'{silhouette(img_path, clusters, data):.2f}')
        plt.savefig(f'cluster{i}.pgf')


if __name__ == "__main__":
    if len(sys.argv) == 3:
        K = sys.argv[1]
        images_dir = Path(sys.argv[2])
    else:
        K = 3
        images_dir = Path("images_cropped")

    random.seed(1234)
    torch.manual_seed(0)
    data = read_data(images_dir)

    print(data['pexels-james-frid-1478419.jpg'][:10])
    


    best_clusters = []
    best_silhuette = -np.inf

    for n in range(100):
        initial_medoids = random.sample(data.keys(), K)
        clusters = k_medoids(data, initial_medoids)

        avg = silhouette_average(data, clusters)
        if avg > best_silhuette:
            best_clusters = clusters
            best_silhuette = avg
    
    print(f"Najboljša povprečna silhueta: {best_silhuette}")

    visualize_clusters(best_clusters)
    