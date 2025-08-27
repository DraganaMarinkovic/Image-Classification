from functools import reduce
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import pickle
import os
from urllib.request import urlretrieve
import tarfile
from collections import defaultdict

from scipy.ndimage import histogram

def download_and_extract_cifar10(root='./data'):
    """
    Download and extract CIFAR-10 dataset if it doesn't exist.
    """
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    if not os.path.exists(root):
        os.makedirs(root)

    filepath = os.path.join(root, filename)

    if not os.path.exists(filepath):
        print("Downloading CIFAR-10...")
        urlretrieve(url, filepath)

    extract_path = os.path.join(root, 'cifar-10-batches-py')
    if not os.path.exists(extract_path):
        print("Extracting files...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(path=root)

    return extract_path


def load_batch(file_path):
    """
    Load a single CIFAR-10 batch file.
    """
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch


def load_balanced_cifar10(samples_per_class=100, root='./data', train=True):
    """
    Load a balanced subset of CIFAR-10 images into a dictionary.

    Args:
        samples_per_class (int): Number of samples to load per class
        root (str): Root directory to store/load CIFAR-10 data
        train (bool): Whether to load from training or test set

    Returns:
        dict: Dictionary with class names as keys and lists of numpy arrays (3x32x32) as values
    """
    # Define the class names
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # Download and extract the dataset if needed
    data_path = download_and_extract_cifar10(root)

    # Initialize dictionary to store images by class
    class_images = defaultdict(list)
    class_counts = {name: 0 for name in class_names}

    if train:
        # Load training batches (1-5)
        batch_files = [f'data_batch_{i}' for i in range(1, 6)]
    else:
        # Load test batch
        batch_files = ['test_batch']

    # Process each batch file
    for batch_file in batch_files:
        batch_path = os.path.join(data_path, batch_file)
        batch = load_batch(batch_path)

        # Get images and labels
        images = batch[b'data']
        labels = batch[b'labels']

        # Process each image
        for img, label in zip(images, labels):
            class_name = class_names[label]

            # If we haven't collected enough samples for this class
            if class_counts[class_name] < samples_per_class:
                # Reshape from (3072) to (32, 32, 3)
                img_reshaped = img.reshape(3, 32, 32).transpose(1, 2, 0)

                # Add to our collection
                class_images[class_name].append(img_reshaped)
                class_counts[class_name] += 1

            # Check if we've collected enough samples for all classes
            if all(count >= samples_per_class for count in class_counts.values()):
                break

        # If we have enough samples, stop processing batches
        if all(count >= samples_per_class for count in class_counts.values()):
            break

    # Convert defaultdict to regular dict
    return dict(class_images)


# Example usage:
"""
# Load 100 images per class from the training set
balanced_cifar = load_balanced_cifar10(samples_per_class=100)

# Access images for a specific class
airplanes = balanced_cifar['airplane']  # List of 100 numpy arrays (3x32x32)

# Print shapes to verify
for class_name, images in balanced_cifar.items():
    print(f"{class_name}: {len(images)} images, shape: {images[0].shape}")
"""

balanced_cifar = load_balanced_cifar10(samples_per_class=100)

# airplane = balanced_cifar['airplane'][1]
#
# #Display the image
# plt.figure(figsize=(3, 3))
# plt.imshow(airplane)
# plt.axis('off')
# plt.show()

#konstanta za broj binova
NUM_BINS = 8

def calculate_normalized_bins_histograms(image_path):
    image = Image.open(image_path)
    print(image)
    #pretvaramo sliku u RGB format(ako nije vec u tom formatu)
    image = image.convert('RGB')

    width, height = image.size

    #dimenzije svakog bina(raspon vrednosti u jednom binu, koliki interval gledamo u odnosu na broj binova)
    bin_size = 256 // NUM_BINS

    #inicijalizujemo histograme za R, G i B komponente
    r_hist = np.zeros(NUM_BINS, dtype=np.float32)
    g_hist = np.zeros(NUM_BINS, dtype=np.float32)
    b_hist = np.zeros(NUM_BINS, dtype=np.float32)

    def pom(coord):
        x, y = coord
        r, g, b = image.getpixel((x, y))

        #izracunavamo indeks odgovarajuceg bina
        r_bin = r // bin_size
        g_bin = g // bin_size
        b_bin = b // bin_size

        return r_bin, g_bin, b_bin

    #kreiramo listu svih koordinata piksela koristeci numpy
    coordinates = np.array(np.meshgrid(np.arange(width), np.arange(height))).T.reshape(-1, 2)
    #kreiramo 3D strukturu,pravi dve 2D matrice kao x y koordinate,transponuje da bi imali (x,y) a ne (y,x), reshape pretvara 3D niz u 2D

    #primenjujemo pom funkciju na svaki piksel koristeci map
    bins = map(pom, coordinates)

    #povecavamo vrednost u odgovarajucem binu
    def update_hist(hist, bin_indices):
        r_bin, g_bin, b_bin = bin_indices

        hist[0][r_bin] += 1
        hist[1][g_bin] += 1
        hist[2][b_bin] += 1
        return hist

    r_hist, g_hist, b_hist = reduce(update_hist, bins, [r_hist, g_hist, b_hist])

    #normalizacija histograma(podela sa ukupnim brojem piksela)
    total_pixels = width * height
    r_hist /= total_pixels
    g_hist /= total_pixels
    b_hist /= total_pixels

    #vracamo rezultat kao numpy matricu
    return np.stack([r_hist, g_hist, b_hist], axis=0)

def calculate_normalized_bins_histograms_from_array(image_array):
    # dimenzije svakog bina(raspon vrednosti u jednom binu, koliki interval gledamo u odnosu na broj binova)
    bin_size = 256 // NUM_BINS

    # inicijalizujemo histograme za R, G i B komponente
    r_hist = np.zeros(NUM_BINS, dtype=np.float32)
    g_hist = np.zeros(NUM_BINS, dtype=np.float32)
    b_hist = np.zeros(NUM_BINS, dtype=np.float32)

    def get_bin_indices(pixel):
        r, g, b = pixel
        r_bin = r // bin_size
        g_bin = g // bin_size
        b_bin = b // bin_size
        return r_bin, g_bin, b_bin

    # pretvaramo image u listu piksela
    pixels = image_array.reshape(-1, 3)

    # mapiramo piksele na njihove odgovarajuce indekse
    bins = map(get_bin_indices, pixels)

    # povecavamo vrednost u odgovarajucem binu
    def update_hist(hist, bin_indices):
        r_bin, g_bin, b_bin = bin_indices
        hist[0][r_bin] += 1
        hist[1][g_bin] += 1
        hist[2][b_bin] += 1
        return hist

    r_hist, g_hist, b_hist = reduce(update_hist, bins, [r_hist, g_hist, b_hist])

    # normalizacija histograma (podela sa ukupnim brojem piksela)
    total_pixels = image_array.size // 3
    r_hist /= total_pixels
    g_hist /= total_pixels
    b_hist /= total_pixels

    return np.stack([r_hist, g_hist, b_hist], axis=0)


def aggregate_histograms(image_list):
    def process_image(acc, item):
        class_name, image = item
        histogram = calculate_normalized_bins_histograms_from_array(image)
        if class_name not in acc:
            acc[class_name] = (histogram, 1)
        else:
            acc[class_name] = (acc[class_name][0] + histogram, acc[class_name][1] + 1)
        return acc

    aggregated = reduce(process_image, image_list, {})

    def average_histogram(acc, class_name):
        histogram, count = aggregated[class_name]
        acc[class_name] = histogram / count
        return acc

    return reduce(average_histogram, aggregated.keys(), {})

def add_class_images_to_list(class_name, num_images, result_list):
    class_images = balanced_cifar[class_name][:num_images]
    new_pairs = map(lambda img: (class_name, np.array(img)), class_images)
    result_list.extend(new_pairs)
    return result_list

#nije potrebno, samo za prikaz
def plot_histograms(histograms, bins_num):
    colors = ['red', 'green', 'blue']  #boje za svaku komponentu
    labels = ['Red', 'Green', 'Blue']  #oznake za komponente

    #petlja kroz tri komponente(R, G, B)
    for i in range(3):
        plt.plot(histograms[i], color=colors[i], label=f'{labels[i]} Component')

    #plt.ylim(0, 1)  #Y-osa od 0 do 1
    plt.title('Normalized Color Histograms')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


    print(f"BINS: ")
    bin_width = 256 // bins_num
    for i in range(bins_num):
        start = i * bin_width
        end = (i + 1) * bin_width
        print(f"\t Bin {i}: {start}-{end}")

    print("Normalized Histograms (RGB):")
    for i in range(3):
        print(f"\t {labels[i]} histogram:  {histograms[i]}")

def cosine_similarity(histogram1, histogram2):

    h1_flat = histogram1.flatten()
    h2_flat = histogram2.flatten()

    # izracunavanje skalarnog proizvoda
    dot_product = reduce(lambda acc, pair: acc + pair[0] * pair[1], zip(h1_flat, h2_flat), 0)

    # izracunavanje norme
    norm1 = reduce(lambda acc, val: acc + val ** 2, h1_flat, 0) ** 0.5
    norm2 = reduce(lambda acc, val: acc + val ** 2, h2_flat, 0) ** 0.5

    # kosinusna slicnost
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0

def classify_image(image_array, class_histograms):

    image_histogram = calculate_normalized_bins_histograms_from_array(image_array)

    # uporedjujemo sliku sa histogramima klasa
    similarities = map(
        lambda class_name: (class_name, cosine_similarity(image_histogram, class_histograms[class_name])),
        class_histograms.keys()
    )

    # trazimo klasu koja je najslicnija
    predicted_class, similarity_score = reduce(
        lambda best, current: current if current[1] > best[1] else best,
        similarities,
        ("", -1)
    )

    return (id(image_array), predicted_class, similarity_score)

if __name__ == '__main__':
    # jedan test sa cifar slikom arrayom i obicna neka sa kompa
    # airplane = balanced_cifar['dog'][9]
    # image_path = 'C:\\Users\\draga\\PycharmProjects\\p24-25-drugi-projekat-tim_draganamarinkovic_dusanjevtic\\img.png'
    # image = Image.open(image_path)
    # image = image.convert('RGB')
    # histogramss = calculate_normalized_bins_histograms_from_array(np.array(image))
    # plot_histograms(histogramss,NUM_BINS)
    # histograms = calculate_normalized_bins_histograms(image_path)
    # plot_histograms(histograms, NUM_BINS)

    # test agregiranih histograma za klase
    list = []
    list = add_class_images_to_list(class_name='dog', num_images=100, result_list=list)
    list = add_class_images_to_list(class_name='cat', num_images=100, result_list=list)
    list = add_class_images_to_list(class_name='deer', num_images=100, result_list=list)
    list = add_class_images_to_list(class_name='bird', num_images=100, result_list=list)
    list = add_class_images_to_list(class_name='horse',num_images=100, result_list=list)

    histograms = aggregate_histograms(list)
    # plot_histograms(histograms['dog'], NUM_BINS)
    # plot_histograms(histograms['cat'], NUM_BINS)

    # kosinusna slicnost
    # print("Testing cosine similarity:")
    # dog_histogram = histograms['dog']
    # cat_histogram = histograms['cat']

    # similarity_score = cosine_similarity(dog_histogram, cat_histogram)
    # print(f"Cosine similarity between 'dog' and 'cat' histograms: {similarity_score}")

    # klasifikacija slike
    print("\nTesting classification:")
    test_image = balanced_cifar['dog'][98]  # Selecting a test image from the 'dog' class
    classification_result = classify_image(np.array(test_image), histograms)
    plot_histograms(calculate_normalized_bins_histograms_from_array(test_image),NUM_BINS)
    print(
        f"Classification Result: Image classified as '{classification_result[1]}' with similarity {classification_result[2]}")
