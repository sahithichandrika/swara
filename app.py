import streamlit as st
from streamlit_option_menu import option_menu
from numpy.core.fromnumeric import prod
import tensorflow as tf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
# from numpy import distutils

# Import the Dataset 
skincare = pd.read_csv("export_skincare.csv", encoding='utf-8', index_col=None)

# Header
st.set_page_config(page_title="Swara", page_icon=":rose:", layout="wide",)

# 1=sidebar menu, 2=horizontal menu, 3=horizontal menu w/ custom menu
EXAMPLE_NO = 2


def streamlit_menu(example=1):
    if example == 1:
        # 1. as sidebar menu
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",  # required
                options=["About Us", "Get Recommendation", "Skin Care "],  # required
                icons=["house", "stars", "book"],  # optional
                menu_icon="cast",  # optional
                default_index=0,  # optional
            )
        return selected

    if example == 2:
        # 2. horizontal menu w/o custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["About Us", "Get Recommendation", "Skin Care "],  # required
            icons=["house", "stars", "book"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
        return selected

    if example == 3:
        # 2. horizontal menu with custom style
        selected = option_menu(
            menu_title=None,  # required
            options=["About Us", "Get Recommendation", "Skin Care "],  # required
            icons=["house", "stars", "book"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {
                    "font-size": "25px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )
        return selected


selected = streamlit_menu(example=EXAMPLE_NO)

if selected == "About Us":
    st.title("Skin Care Product Recommender :sparkles:")
    st.write('---') 

    st.write(
        """
        ##### 

"**Skin Care Product Recommendation Application is an implementation of Machine Learning that can provide recommendations for skin care products according to your skin type and issues.**"
        """)
    
    #displaying a local video file

    video_file = open("skincare.mp4", "rb").read()
    st.video(video_file, start_time = 1) #displaying the video 
    
    st.write(' ') 
    st.write(' ')
    st.write(
        """
        ##### You will receive skin care product recommendations from a variety of cosmetic brands,  tailored to your skin needs. 
        ##### There are 5 categories of skin care products for 5 different skin types, as well as concerns and benefits you wish to achieve from the products. This recommendation application is merely a system that provides suggestions based on the data you input, not scientific consultation.
        ##### Please select the Get Recommendation page to start receiving recommendations, or choose the Skin Care  page to view tips and tricks about skin care.
        """)
    
    st.write(
        """
        **Happy Trying! :) !**
        """)
    
    
    st.info('Credit: Created by Team SWARA')

if selected == "Get Recommendation":
    st.title(f"Let's {selected}")
    
    st.write(
        """
        ##### **To get recommendations, please enter your skin type, concerns, and desired benefits to receive suitable skin care product suggestions.**
        """) 
    
    st.write('---') 

    first, last = st.columns(2)

    # Choose a product product type category
    # pt = product type
    category = first.selectbox(label='Product Category : ', options=skincare['product_type'].unique())
    category_pt = skincare[skincare['product_type'] == category]

    # Choose a skin type
    # st = skin type
    skin_type = last.selectbox(label='Your Skin Type : ', options=['Normal', 'Dry', 'Oily', 'Combination', 'Sensitive'])
    category_st_pt = category_pt[category_pt[skin_type] == 1]

    # Choose concerns
    prob = st.multiselect(label='Skin Problems : ', options=['Dull Skin', 'Acne', 'Acne Scars', 'Large Pores', 'Dark Spots', 'Fine Lines and Wrinkles', 'Blackheads', 'Uneven Skin Tone', 'Redness', 'Sagging Skin'])

    # Choose notable effects
    opsi_ne = category_st_pt['notable_effects'].unique().tolist()
    selected_options = st.multiselect('Notable Effects : ', opsi_ne)
    category_ne_st_pt = category_st_pt[category_st_pt["notable_effects"].isin(selected_options)]

    # Choose product
    opsi_pn = category_ne_st_pt['product_name'].unique().tolist()
    product = st.selectbox(label='Recommended Product For You', options=sorted(opsi_pn))

    ## MODELLING with Content Based Filtering
    tf = TfidfVectorizer()
    tf.fit(skincare['notable_effects']) 
    tfidf_matrix = tf.fit_transform(skincare['notable_effects']) 
    shape = tfidf_matrix.shape
    tfidf_matrix.todense()

    pd.DataFrame(
        tfidf_matrix.todense(), 
        columns=tf.get_feature_names_out(),
        index=skincare.product_name
    ).sample(shape[1], axis=1).sample(10, axis=0)

    cosine_sim = cosine_similarity(tfidf_matrix) 
    cosine_sim_df = pd.DataFrame(cosine_sim, index=skincare['product_name'], columns=skincare['product_name'])

    def skincare_recommendations(nama_produk, similarity_data=cosine_sim_df, items=skincare[['product_name', 'brand', 'description']], k=5):
        index = similarity_data.loc[:,nama_produk].to_numpy().argpartition(range(-1, -k, -1))
        closest = similarity_data.columns[index[-1:-(k+2):-1]]
        closest = closest.drop(nama_produk, errors='ignore')
        df = pd.DataFrame(closest).merge(items).head(k)
        return df

    model_run = st.button('Find More Product Recommendations!')
    if model_run:
        st.write('"Here are other similar product recommendations based on your preferences"')
        st.write(skincare_recommendations(product))

    
    
if selected == "Skin Care ":
    st.title(f"Take a Look at {selected}")
    st.write('---') 

    st.write(
        """
        ##### **"Here are tips and tricks you can follow to maximize the use of your skincare products"**
        """) 
    
    image = Image.open('imagepic.jpg')
    st.image(image, caption='Skin Care 101')
    

    
    st.write(
        """
        ### **1. Face Wash**
        """)
    st.write(
        """
        **- "Use the recommended face wash products or those that are suitable for you"**
        """)
    st.write(
        """
        **- Wash your face a maximum of twice a day: in the morning and at night before bed. Washing your face too often can strip away the skin's natural oils. If you have dry skin, it's okay to use just water in the morning.**
        """)
    st.write(
        """
        **- Avoid scrubbing your face harshly as it can remove the skin's natural barrier.**
        """)

    st.write(
        """
        **- The best way to cleanse your skin is by using your fingertips for 30-60 seconds with circular motions and gentle massage.**
        """)

    
    st.write(
        """
        ### **2. Toner**
        """)
    st.write(
        """
        **- Use a toner that has been recommended or one that is suitable for you.**
        """)
    st.write(
        """
        **- Pour the toner onto a cotton pad and gently apply it to your face. For better results, use two layers of toner, first with a cotton pad and the last with your hands for better absorption.**
        """)
    st.write(
        """
        **- Apply toner after washing your face.**
        """)

    st.write(
        """
        **- If you have sensitive skin, avoid skincare products that contain fragrance as much as possible.**
        """)

    
    st.write(
        """
        ### **3. Serum**
        """)
    st.write(
        """
        **- Use a serum that has been recommended or one that is suitable for you for better results.**
        """)
    st.write(
        """
        **- Apply serum after your face is completely clean to ensure the serum absorbs effectively.**
        """)
    st.write(
        """
        **- Use serum in the morning and at night before bed.**
        """)
    st.write(
        """
        **- Choose a serum based on your needs, such as for acne scars, dark spots, anti-aging, or other benefits.**
        """)

    st.write(
        """
        **- To ensure the serum absorbs better, pour it into your palm, gently pat it onto your face, and wait until it absorbs.**
        """)

    
    st.write(
        """
        ### **4. Moisturizer**
        """)
    st.write(
        """
        **- Use a moisturizer that has been recommended or one that is suitable for you for better results.**
        """)
    st.write(
        """
        **- Moisturizer is an essential skincare product that locks in moisture and nutrients from the serum you have used.**
        """)
    st.write(
        """
        **- For optimal results, use different moisturizers in the morning and at night. Morning moisturizers usually contain sunscreen and vitamins to protect the skin from UV damage and pollution, while night moisturizers include various active ingredients that support skin regeneration while you sleep.**
        """)
    st.write(
        """
        **- Allow a 2-3 minute gap between applying serum and moisturizer to ensure the serum has fully absorbed into the skin.**
        """)

    st.write(
        """
        ### **5. Sunscreen**
        """)
    st.write(
        """
        **- Use a sunscreen that has been recommended or one that is suitable for you for better results.**
        """)
    st.write(
        """
        **- Sunscreen is a key product in any skincare routine because it protects the skin from the harmful effects of UVA and UVB rays, and even blue light. All skincare benefits are diminished without proper protection.**
        """)
    st.write(
        """
        **- Apply sunscreen approximately the size of your index and middle finger to ensure maximum protection.**
        """)
    st.write(
        """
        **- Reapply sunscreen every 2-3 hours or as needed.**
        """)

    st.write(
        """
        **- Continue using sunscreen even when indoors, as sunlight can still penetrate through windows after 10 AM and during cloudy weather.**
        """)
    
    st.write(
        """
        ### **6. Avoid Switching Skincare Products Frequently**
        """)
    st.write(
        """
        **Frequently changing skincare products can cause skin stress as it has to adapt to different ingredients. As a result, the benefits may not be fully realized. Instead, use skincare products consistently for several months to see the results.**
        """)

    
    st.write(
        """
        ### **7. Consistency**
        """)
    st.write(
        """
        **The key to facial care is consistency. Be diligent and persistent in using skincare products, as the results are not instant.**
        """)
    st.write(
        """
        ### **8. The Face is an Asset**
        """)
    st.write(
        """
        **Human diversity is a blessing given by the Creator. Take good care of this blessing with gratitude. Choose products and methods that suit your skin's needs. Using skincare products early on is like investing in your future.**
        """)

     
    
    
