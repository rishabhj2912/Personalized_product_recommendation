from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import keras
from keras.models import load_model
import pandas as pd

# Load the brands CSV
brands_df = pd.read_csv('product.csv')
b_df=pd.read_csv('brand.csv')

# Create a dictionary for easy brand lookup
brand_lookup = {}
for index, row in brands_df.iterrows():
    brand_lookup[row['product_id']] = {
        'brand': row['brand'],
        'brand_id': row['brand_id'],
        'rating':row['rating']
    }

b_lookup = {}
for index, row in b_df.iterrows():
    b_lookup[row['brand_id']] = {
        'brand': row['brand']
    }
app = Flask(__name__)

# Load the saved model
# model = load_model('model.h5')
# model2= load_model('model2.h5')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search',methods=['POST'])
def search():
    brand = request.form['brand']
    user_id = request.form['user_id']

    from sklearn.preprocessing import LabelEncoder
    from Neural import user_enc, n_products, item_enc, model
    from Neural2 import user_enc2, n_brands, brand_enc, model2

    target_user_encoded = user_enc.transform([user_id])[0]
    target_user_encoded2 = user_enc2.transform([user_id])[0]
    candidate_product_ids = np.arange(n_products)
    candidate_brand_ids=np.arange(n_brands)  # n_products needs to be defined in your code

    
    predicted_purchase_counts = model.predict([np.array([target_user_encoded] * len(candidate_product_ids)), candidate_product_ids])
    predicted_brand_counts= model2.predict([np.array([target_user_encoded2] * len(candidate_brand_ids)), candidate_brand_ids])

    # Sort recommendations
    recommended_product_indices = np.argsort(predicted_purchase_counts, axis=0)[::-1]
    recommended_product_ids = item_enc.inverse_transform(candidate_product_ids[recommended_product_indices])
    recommended_products_with_brand = [
        {
            'product_id': product_id,
            'brand': brand_lookup.get(product_id, {}).get('brand', 'N/A'),
            'brand_id': brand_lookup.get(product_id, {}).get('brand_id', 'N/A'),
            'rating': brand_lookup.get(product_id, {}).get('rating', 'N/A')

        }
        for product_id in recommended_product_ids
    ]
    filtered_products = [
        product for product in recommended_products_with_brand
        if product['brand'] == brand
    ]

    return render_template('search.html', user_id=brand, search=filtered_products[:10])


@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.form['user_id']
    
    from sklearn.preprocessing import LabelEncoder
    from Neural import user_enc, n_products, item_enc, model
    from Neural2 import user_enc2, n_brands, brand_enc, model2

    target_user_encoded = user_enc.transform([user_id])[0]
    target_user_encoded2 = user_enc2.transform([user_id])[0]
    candidate_product_ids = np.arange(n_products)
    candidate_brand_ids=np.arange(n_brands)  # n_products needs to be defined in your code

    
    predicted_purchase_counts = model.predict([np.array([target_user_encoded] * len(candidate_product_ids)), candidate_product_ids])
    predicted_brand_counts= model2.predict([np.array([target_user_encoded2] * len(candidate_brand_ids)), candidate_brand_ids])

    # Sort recommendations
    recommended_product_indices = np.argsort(predicted_purchase_counts, axis=0)[::-1]
    recommended_product_ids = item_enc.inverse_transform(candidate_product_ids[recommended_product_indices])
    recommended_products_with_brand = [
        {
            'product_id': product_id,
            'brand': brand_lookup.get(product_id, {}).get('brand', 'N/A'),
            'brand_id': brand_lookup.get(product_id, {}).get('brand_id', 'N/A'),
            'rating': brand_lookup.get(product_id, {}).get('rating', 'N/A')

        }
        for product_id in recommended_product_ids
    ]
    recommended_brand_indices = np.argsort(predicted_brand_counts, axis=0)[::-1]
    recommended_brand_ids = brand_enc.inverse_transform(candidate_brand_ids[recommended_brand_indices])
    brand_recommendations = [
        {
            'brand_id': brand_id,
            'brand': b_lookup.get(brand_id, {}).get('brand', 'N/A')
        }
        for brand_id in recommended_brand_ids
    ]
    

    # Render the template with recommendations
    return render_template('recommendations.html', user_id=user_id, recommendations=recommended_products_with_brand[:10],recommendations2=brand_recommendations[:10])

if __name__ == '__main__':
    app.run(debug=True)
