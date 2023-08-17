from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


def load_and_preprocess_data(filename):
    dateparse = lambda x: pd.to_datetime(x, format='%m/%d/%Y %H:%M')
    df = pd.read_csv(filename, parse_dates=['InvoiceDate'], date_parser=dateparse, encoding='unicode_escape')
    df = df.loc[df['Quantity'] > 0]
    df1 = df.dropna(subset=['CustomerID'])
    
    return df1

def generate_item_recommendations(customer_id, customer_item_matrix, df):
    items_bought_by_A = set(customer_item_matrix.loc[customer_id][customer_item_matrix.loc[customer_id] > 0].index)
    
    items_to_recommend_User_B = items_bought_by_A - items_bought_by_B
    
    recommendations = df.loc[
        df['StockCode'].isin(items_to_recommend_User_B),
        ['StockCode', 'Description']
    ].drop_duplicates().set_index('StockCode')
    
    return recommendations

df = load_and_preprocess_data('OnlineRetail.csv')

customer_item_matrix = df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', aggfunc='sum')
customer_item_matrix = customer_item_matrix.applymap(lambda x: 1 if x > 0 else 0)

item_item_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix.T))
item_item_sim_matrix.columns = customer_item_matrix.T.index
item_item_sim_matrix['StockCode'] = customer_item_matrix.T.index
item_item_sim_matrix = item_item_sim_matrix.set_index('StockCode')

item_id = '22633'

top_10_similar_items = list(
    item_item_sim_matrix
    .loc[item_id]
    .sort_values(ascending=False)
    .iloc[:10]
    .index
)

recommendations_item_item = df.loc[
    df['StockCode'].isin(top_10_similar_items),
    ['StockCode', 'Description']
].drop_duplicates().set_index('StockCode').loc[top_10_similar_items]

print("RECOMMENDATIONS#2")
print(recommendations_item_item)


user_to_user_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix))
user_to_user_sim_matrix.columns = customer_item_matrix.index
user_to_user_sim_matrix['CustomerID'] = customer_item_matrix.index
user_to_user_sim_matrix = user_to_user_sim_matrix.set_index('CustomerID')

customer_id_B = 17935.0

most_similar_customer = user_to_user_sim_matrix.loc[customer_id_B].sort_values(ascending=False).index[1]

items_bought_by_A = set(customer_item_matrix.loc[most_similar_customer][customer_item_matrix.loc[most_similar_customer] > 0].index)
items_bought_by_B = set(customer_item_matrix.loc[customer_id_B][customer_item_matrix.loc[customer_id_B] > 0].index)

items_to_recommend_User_B = items_bought_by_A - items_bought_by_B

recommendations = df.loc[
    df['StockCode'].isin(items_to_recommend_User_B),
    ['StockCode', 'Description']
].drop_duplicates().set_index('StockCode')

print(f"RECOMMENDATIONS for Customer {customer_id_B} based on Customer {most_similar_customer}")
print(recommendations)

#take all the past purchases of a userid and genrate items that are similar to those items for each item

customer_id=17935.0
# get all items bought by this customer
items_bought_by_A = set(customer_item_matrix.loc[customer_id][customer_item_matrix.loc[customer_id] > 0].index)
#generate similar items for each item bought
similar_items_to_buy = pd.DataFrame()
for item in items_bought_by_A:
    similar_items = pd.DataFrame(item_item_sim_matrix.loc[item].sort_values(ascending=False).iloc[:10]).reset_index()
    similar_items.columns = ['StockCode', 'SimilarityIndex']
    similar_items['CustomerID'] = customer_id
    similar_items_to_buy = pd.concat([similar_items_to_buy, similar_items], ignore_index=True)

# Print items similar to each item bought by the customer
print(similar_items_to_buy)



