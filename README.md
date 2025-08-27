# Project: Color Histogram Classification  

## Overview  
This project implements a simple **image classification system** based on **color histograms** in the RGB color space.  
The workflow:  
1. Compute histograms for each image.  
2. Compute average histograms per class.  
3. Compare histograms using **cosine similarity**.  
4. Implement a naive classifier that assigns images to the class with the highest similarity.  

---

## Constraints  
- All code must be in a **single `.py` file**.  
- **Not allowed**: explicit loops (`for`, `while`), list/dict comprehensions, `len()`.  
- Allowed:  
  - `map`, `reduce`, `sorted`, `lambda`  
  - `list()` to force iterators (e.g., after `map`)  
  - `numpy.flatten()` for converting matrices to vectors  
- Implement `len` via `reduce` if needed.  

---

## Tasks  

### 1. Histogram Computation (5 pts)  
- Function input: path to an image.  
- Output: one histogram for each RGB component.  
- Bin count defined as a global constant (suggested: 8–16).  
- Each histogram is normalized by total number of pixels.  
- Result: **2D numpy matrix**, each dimension corresponds to bins for R/G/B values.  

### 2. Average Histogram per Class (7 pts)  
- Input: list of `(class, image_path)` pairs.  
- Compute aggregated histogram for each class using `map` / `reduce`.  
- Divide by number of images in the class to get average.  
- Result: list of tuples `(class, average_histogram)` where histogram is a 2D numpy matrix.  

### 3. Cosine Similarity (2 pts)  
- Compute cosine similarity between two histograms.  
- Steps:  
  - Flatten histograms into 1D arrays.  
  - Use `map` / `reduce` to compute dot product and norms.  
- Result: a single numeric similarity score.  

### 4. Classifier (6 pts)  
- For each test image:  
  1. Compute its histogram.  
  2. Compare with average histograms of all classes using cosine similarity.  
  3. Assign to class with **highest similarity**.  
- Result: tuple `(image_id, predicted_class, similarity_value)`.  

---

## Datasets    
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)   

---

⚠️ Note: This classification approach is very naive and will likely not yield high accuracy. The purpose is to **practice the required functional programming style** in Python.  

