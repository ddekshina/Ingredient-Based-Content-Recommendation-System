# Ingredient-Based Content Recommendation System

## Introduction

This project implements a content-based recommendation system for skincare products, specifically focusing on moisturizers designed for dry skin. The system utilizes ingredient information to recommend similar products to a user's input. The project also includes interactive visualizations of product clusters based on ingredient similarity.

## Problem Statement / Idea

The project aims to address the challenge of helping users discover skincare products based on ingredient compatibility. Users can input a product and receive recommendations for similar products. The interactive visualizations allow for exploration of product clusters based on ingredient analysis.

## Features

*   **Data Processing:** Loads, cleans, and preprocesses Sephora cosmetics dataset.
*   **Feature Engineering:** Employs TF-IDF vectorization for ingredient-based feature representation.
*   **Dimensionality Reduction:** Uses t-SNE for dimensionality reduction and visualization of product clusters.
*   **Interactive Visualization:** Generates an interactive Bokeh plot to visualize product clusters based on ingredient similarity. Allows hovering over data points to display product details, as well as panning, zooming, and saving the plot.
*   **Recommendation Engine:**  Calculates cosine similarity to recommend similar products based on ingredient profiles. Includes a filter to avoid recommending simple product variations.
*   **User Interaction:** Provides a command-line interface to input a product name and receive recommendations, and displays shared ingredients to justify the recommendation.

## How It Works (Implementation Overview)

1.  **Data Loading & Preprocessing:** The system begins by loading a dataset containing cosmetic product information (likely from `cosmetics.csv`). Data cleaning and preprocessing is performed to prepare the data for analysis.
2.  **Feature Extraction:**  TF-IDF vectorization is applied to extract features based on product ingredients.
3.  **Dimensionality Reduction:** t-SNE is used to reduce the dimensionality of the ingredient vectors, enabling visualization of product clusters.
4.  **Interactive Visualization Generation:** A Bokeh plot is generated that visualizes the product clusters in an interactive format.
5.  **Recommendation Calculation:** Cosine similarity is computed between product ingredient vectors to identify the most similar products.
6.  **Recommendation Display:** The system outputs product recommendations, along with a similarity score and shared ingredient information.
7.  **User Interface:** A command-line interface allows users to input a product name and see the recommended products.

## Tech Stack

*   **Programming Language:** Python
*   **Libraries:**
    *   pandas: Data manipulation and analysis.
    *   re: Regular expressions for text processing.
    *   sklearn (scikit-learn): `TfidfVectorizer`, `TSNE`, `cosine_similarity`.
    *   bokeh: Interactive visualization.
*   **Dependencies:** The dependencies are listed in `requirements.txt`: pandas, scikit-learn, bokeh.

## Getting Started

The project requires Python and the libraries listed in `requirements.txt`. To run the script, install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

The script expects a CSV file named `cosmetics.csv` in the same directory. To use the visualization, the generated `product_clusters.html` can be opened in a web browser.

The project appears to also utilize a gitignore file to exclude a "myenv" folder, which may indicate the project uses a virtual environment for dependency management.

## Conclusion

This project delivers a content-based recommendation system capable of providing personalized skincare product recommendations based on ingredient compatibility. It includes an interactive visualization for exploring product clusters based on ingredient similarity, and allows users to input a product name and receive recommendations along with a justification. This system can be used by customers to help them discover new skincare products based on their input.
```
