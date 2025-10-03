# Ingredient-Based Skincare Recommendation System
# This script builds a content-based recommendation system for moisturizers
# suitable for dry skin, using the Sephora cosmetics dataset.

# --- 1. Import Libraries ---
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.io import output_file, save

# --- 2. Data Loading and Preprocessing ---
def load_and_preprocess_data(filepath):
    """
    Loads the dataset, filters for moisturizers for dry skin,
    and performs necessary cleaning and tokenization.
    """
    print("Loading and preprocessing data...")
    # Load the dataset
    df = pd.read_csv(filepath)

    # Filter for moisturizers suitable for dry skin
    moisturizers = df[(df['Label'] == 'Moisturizer') & (df['Dry'] == 1)].copy()

    # Data Cleaning: Handle missing/invalid ingredient data
    moisturizers = moisturizers[moisturizers['Ingredients'] != '#NAME?']
    moisturizers = moisturizers[~moisturizers['Ingredients'].str.contains("Visit the", na=False)]
    moisturizers.dropna(subset=['Ingredients'], inplace=True)
    moisturizers.reset_index(drop=True, inplace=True)

    # Standardize text to lowercase
    for col in ['Brand', 'Name', 'Ingredients']:
        moisturizers[col] = moisturizers[col].str.lower()

    # Remove duplicates
    moisturizers.drop_duplicates(subset=['Brand', 'Name'], inplace=True)
    moisturizers.reset_index(drop=True, inplace=True)

    # Ingredient Tokenization
    def tokenize_ingredients(ingredient_str):
        # Remove parentheses and their contents, split by comma, and clean up
        ingredients = re.sub(r'\s*\(.*\)\s*', '', ingredient_str)
        tokens = [ing.strip() for ing in ingredients.split(',')]
        return [token for token in tokens if token] # Remove any empty strings

    moisturizers['ingredients_list'] = moisturizers['Ingredients'].apply(tokenize_ingredients)
    
    print(f"Preprocessing complete. Found {len(moisturizers)} moisturizers for dry skin.")
    return moisturizers

# --- 3. Feature Engineering ---
def create_ingredient_matrix(df):
    """
    Creates a document-term matrix (TF-IDF) from the ingredient lists.
    """
    print("Creating TF-IDF ingredient matrix...")
    # Use TF-IDF to give more weight to rare, defining ingredients
    # The tokenizer expects a string, so we join the list back
    df['ingredient_str'] = df['ingredients_list'].apply(lambda x: ' '.join(x).replace('*','')) # Clean up ingredient strings
    
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['ingredient_str'])
    
    print(f"TF-IDF matrix created with shape: {tfidf_matrix.shape}")
    return tfidf_matrix, vectorizer

# --- 4. Dimensionality Reduction ---
def reduce_dimensions(matrix, df):
    """
    Applies t-SNE to reduce the ingredient matrix to 2 dimensions for plotting.
    """
    print("Applying t-SNE for dimensionality reduction...")
    # Note: 'n_iter' is deprecated in newer scikit-learn versions. Using 'max_iter' instead.
    # Added init='pca' and learning_rate='auto' for better performance and stability with newer versions.
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(matrix.toarray())
    
    df['tsne_x'] = tsne_results[:, 0]
    df['tsne_y'] = tsne_results[:, 1]
    
    print("t-SNE complete.")
    return df

# --- 5. Interactive Visualization ---
def create_interactive_plot(df, output_filename="product_clusters.html"):
    """
    Creates an interactive Bokeh scatter plot to visualize product clusters.
    """
    print(f"Creating interactive Bokeh plot, will be saved to '{output_filename}'...")
    # Create a short version of ingredients for the tooltip
    df['ingredients_short'] = df['ingredients_list'].apply(lambda x: ', '.join(x[:5]) + '...')

    # Create a Bokeh ColumnDataSource
    source = ColumnDataSource(data=dict(
        x=df['tsne_x'],
        y=df['tsne_y'],
        name=df['Name'].str.title(),
        brand=df['Brand'].str.title(),
        price=df['Price'],
        ingredients=df['ingredients_short']
    ))

    # Define the hover tooltips
    hover = HoverTool(tooltips=[
        ("Product", "@name"),
        ("Brand", "@brand"),
        ("Price", "$@price"),
        ("Key Ingredients", "@ingredients")
    ])

    # Create the plot
    p = figure(
        width=1000,
        height=700,
        tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save'],
        title="Ingredient Similarity of Moisturizers for Dry Skin (t-SNE)"
    )
    p.title.align = 'center'
    p.title.text_font_size = '1.5em'
    
    p.scatter(
        x='x',
        y='y',
        source=source,
        size=10,
        alpha=0.7,
        legend_label="Moisturizers"
    )
    p.legend.location = "top_left"
    p.xaxis.axis_label = "t-SNE Component 1"
    p.yaxis.axis_label = "t-SNE Component 2"

    # Save the plot to an HTML file
    output_file(output_filename, title="Skincare Product Clusters")
    save(p)
    print("Plot saved successfully.")

# --- 6. Recommendation Engine ---
def get_recommendations(product_name, df, matrix):
    """
    Finds the top 5 most similar products based on cosine similarity
    of their ingredient vectors. Filters out simple size variations.
    """
    # Normalize product name for matching
    product_name = product_name.lower()
    
    if product_name not in df['Name'].values:
        return f"Product '{product_name.title()}' not found in the dataset.", None, None

    # Get the index of the input product
    idx = df[df['Name'] == product_name].index[0]

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(matrix[idx], matrix)

    # Get similarity scores and sort them
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # --- IMPROVEMENT: Filter out simple variations of the same product ---
    # Get the base name of the product to filter out "mini" or "limited edition" versions.
    base_name = product_name.replace(' mini', '').replace(' limited-edition', '')
    
    # Get the top N most similar product indices, skipping the original product itself
    # and any simple variations of it.
    recommended_indices = []
    scores = []
    for i, score in sim_scores:
        if len(recommended_indices) >= 5:
            break
        # Skip the original product
        if i == idx:
            continue
        # Skip products that are just size variations
        if base_name in df.iloc[i]['Name']:
            continue
        recommended_indices.append(i)
        scores.append(score)

    recommendations = df.iloc[recommended_indices].copy()
    
    # Add similarity score to the output
    recommendations['similarity_score'] = scores

    return recommendations[['Brand', 'Name', 'Price', 'similarity_score']], df.loc[idx, 'ingredients_list']

# --- Main Execution Block ---
if __name__ == "__main__":
    # Define the file path for the dataset
    DATA_FILE = 'cosmetics.csv'
    
    # Step 1: Load and preprocess the data
    moisturizers_df = load_and_preprocess_data(DATA_FILE)
    
    # Step 2: Create the ingredient matrix
    tfidf_matrix, vectorizer = create_ingredient_matrix(moisturizers_df)
    
    # Step 3: Reduce dimensions for visualization
    moisturizers_df_tsne = reduce_dimensions(tfidf_matrix, moisturizers_df)
    
    # Step 4: Create and save the interactive plot
    create_interactive_plot(moisturizers_df_tsne)
    
    # Step 5: Demonstrate the recommendation engine
    print("\n--- Recommendation Engine Demo ---")
    sample_product = 'crème de la mer'
    recommendations, original_ingredients = get_recommendations(sample_product, moisturizers_df, tfidf_matrix)

    if isinstance(recommendations, pd.DataFrame):
        print(f"Top 5 recommendations for '{sample_product.title()}':")
        print(recommendations.to_string())

        # --- IMPROVED JUSTIFICATION ---
        # Get the top recommendation for a more detailed and accurate justification
        top_rec_row = recommendations.iloc[0]
        top_rec_ingredients = moisturizers_df[moisturizers_df['Name'] == top_rec_row['Name'].lower()]['ingredients_list'].values[0]
        
        # Compare the original, full ingredient lists to get a true count of shared ingredients
        shared_ingredients = set(original_ingredients) & set(top_rec_ingredients)
        
        print("\n--- Justification for Top Recommendation ---")
        print(f"Product: {top_rec_row['Name'].title()} by {top_rec_row['Brand'].title()}")
        print(f"Similarity Score: {top_rec_row['similarity_score']:.2f}")
        print(f"Based on the full ingredient lists, it shares {len(shared_ingredients)} ingredients with '{sample_product.title()}'.")
        print(f"Key shared ingredients include: {', '.join(list(shared_ingredients)[:5])}...")
    else:
        print(recommendations)

