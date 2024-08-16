ortfrom numpy.core.fromnumeric import prod
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Import the Dataset 
skincare = pd.read_csv("export_skincare.csv", encoding='utf-8', index_col=None)

# Header
st.set_page_config(page_title="Swara", page_icon=":blossom:", layout="wide")

# display the main page
st.title("Swara :sparkles:")

st.write('---') 

#displaying a local video file

video_file = open("skincare.mp4", "rb").read()
st.video(video_file, start_time = 1) #displaying the video 


st.write('---') 

st.write(
    """
    ##### **The Skincare Product Recommendation Application is an implementation of a Machine Learning project that provides skincare product recommendations based on your skin type and concerns. You can input your skin type, issues, and desired benefits to receive accurate skincare product recommendations.**
    """)

st.write('---') 

first,last = st.columns(2)

# Choose a product product type category
# pt = product type
category = first.selectbox(label='Product Category : ', options= skincare['product_type'].unique() )
category_pt = skincare[skincare['product_type'] == category]

# Choose a skintype
# st = skin type
skin_type = last.selectbox(label='Your Skin Type : ', options= ['Normal', 'Dry', 'Oily', 'Combination', 'Sensitive'] )
category_st_pt = category_pt[category_pt[skin_type] == 1]

# Choose skin concerns
prob = st.multiselect(
    label='Your Skin Problems : ',
    options=['Dull Skin', 'Acne', 'Acne Scars', 'Large Pores', 'Dark Spots', 'Fine Lines and Wrinkles', 'Blackheads', 'Uneven Skin Tone', 'Redness', 'Sagging Skin']
)

# Choose notable_effects
# From the filtered products based on product type and skin type (category_st_pt), get unique values in the 'notable_effects' column
options_ne = category_st_pt['notable_effects'].unique().tolist()

# 'notable_effects' unique values are assigned to the variable options_ne and used as values in the multiselect wrapped in the selected_options variable below
selected_options = st.multiselect('Manfaat yang Diinginkan : ', options_ne)

# Filter the dataframe based on the selected notable effects
category_ne_st_pt = category_st_pt[category_st_pt["notable_effects"].isin(selected_options)]


# Choose product
# Filtered products in the variable filtered_df, then extract unique product names and store them in the variable opsi_pn
options_pn = category_ne_st_pt['product_name'].unique().tolist()

# Create a selectbox with the filtered product options
product = st.selectbox(
    label='Products Recommended for You',
    options=sorted(options_pn)
)

# The variable 'product' will hold a selected product that will trigger the display of other product recommendations

## MODELLING with Content Based Filtering
# Initiate TfidfVectorizer
tf = TfidfVectorizer()

# Perform IDF calculation on the 'notable_effects' data
tf.fit(skincare['notable_effects'])

# Mapping from feature index integers to feature names
feature_names = tf.get_feature_names()

# Fit and transform the data into a TF-IDF matrix
tfidf_matrix = tf.fit_transform(skincare['notable_effects'])


# Check the size of the TF-IDF matrix
shape = tfidf_matrix.shape

# Convert the TF-IDF vector to a dense matrix
dense_matrix = tfidf_matrix.todense()

# Create a DataFrame to view the TF-IDF matrix
# Columns are filled with the notable effects
# Rows are filled with product names
df_tfidf = pd.DataFrame(
    dense_matrix, 
    columns=tf.get_feature_names(),
    index=skincare['product_name']
)

# Display a sample of the TF-IDF matrix
df_tfidf.sample(shape[1], axis=1).sample(10, axis=0)

# Calculate cosine similarity on the TF-IDF matrix
cosine_sim = cosine_similarity(tfidf_matrix)

# Create a DataFrame from the cosine similarity matrix with rows and columns as product names
cosine_sim_df = pd.DataFrame(
    cosine_sim, 
    index=skincare['product_name'], 
    columns=skincare['product_name']
)

# View a sample of the similarity matrix for each product name
cosine_sim_df.sample(7, axis=1).sample(10, axis=0)

def skincare_recommendations(product_name, similarity_data=cosine_sim_df, items=skincare[['product_name', 'product-href', 'price', 'description']], k=5):
    """
    Get skincare product recommendations based on the similarity matrix.

    :param product_name: The name of the product for which recommendations are needed
    :param similarity_data: DataFrame containing similarity scores between products
    :param items: DataFrame containing product details
    :param k: Number of recommendations to return
    :return: DataFrame with recommended products
    """
    # Use argpartition to indirectly partition along the given axis
    # Convert DataFrame to numpy array and get indices for top k similar products
    index = similarity_data.loc[:, product_name].to_numpy().argpartition(range(-1, -k, -1))
    
    # Get the product names with the highest similarity
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    
    # Drop the searched product from the recommendations
    closest = closest.drop(product_name, errors='ignore')
    
    # Merge with the items DataFrame to get product details and return top k recommendations
    df = pd.DataFrame(closest, columns=['product_name']).merge(items, on='product_name').head(k)
    
    return df


# Create a button to display recommendations
model_run = st.button('Find More Product Recommendations!')

# Get recommendations when the button is clicked
if model_run:
    st.write('"Here are other similar product recommendations based on your preferences"')
    st.write(skincare_recommendations(product))
    st.snow()

