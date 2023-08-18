import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import train_test_split, precision_at_k, AUC_at_k, mean_average_precision_at_k, ndcg_at_k
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid

def load_and_preprocess_data(filename):
    dateparse = lambda x: pd.to_datetime(x, format='%m/%d/%Y %H:%M')
    df = pd.read_csv(filename, parse_dates=['InvoiceDate'], date_parser=dateparse, encoding='unicode_escape')
    df = df.loc[df['Quantity'] > 0]
    df1 = df.dropna(subset=['CustomerID'])
    return df1
df = load_and_preprocess_data('OnlineRetail.csv')
#remove columns with missing customerID
df = df.dropna(subset=['CustomerID'])
#remove columns with missing description
df = df.dropna(subset=['Description'])
#remove columns with missing stockcode
df = df.dropna(subset=['StockCode'])
stock_code_encoder = LabelEncoder()
df['StockCodeEncoded'] = stock_code_encoder.fit_transform(df['StockCode'])
df = df.drop_duplicates(subset='StockCodeEncoded')
has_duplicates = df['StockCodeEncoded'].duplicated().any()
if has_duplicates:
    print("There are duplicate encoded StockCode values.")
    df['StockCodeEncoded'] = stock_code_encoder.fit_transform(df['StockCode'])

    
max_encoded_value = df['StockCodeEncoded'].max()

if max_encoded_value >= 3665:
    print("Encoded values exceed the valid range.")
inconsistent_encoding = df.groupby('StockCode')['StockCodeEncoded'].nunique() > 1
if inconsistent_encoding.any():
    print("Inconsistent encoding found.")
unique_stock_code_count = df.groupby('StockCode')['StockCodeEncoded'].nunique()
if unique_stock_code_count.max() > 1:
    print("Some StockCode values have inconsistent encoding.")


interaction_matrix = coo_matrix((df['Quantity'], (df['CustomerID'].astype(int), df['StockCodeEncoded'])))
train, test = train_test_split(interaction_matrix, train_percentage=0.8)

train = train.tocsr()
test = test.tocsr()
param_grid = {
    'factors': [20, 50, 100],
    'regularization': [0.01, 0.1, 1.0],
    'iterations': [10, 20, 30]
}
best_precision = 0
best_params = {'factors': 50,
    'regularization': 0.01,
    'iterations': 10}

for params in ParameterGrid(param_grid):
    model = AlternatingLeastSquares(factors=params['factors'], regularization=params['regularization'], iterations=params['iterations'], use_gpu=True)
    model.fit(train)

    precision = precision_at_k(model, train, test, K=5)
    
    if precision > best_precision:
        best_precision = precision
        best_params['factors'] = params['factors']
        best_params['regularization'] = params['regularization']
        best_params['iterations'] = params['iterations']

print(f'Best Precision@5: {best_precision:.4f}')
print(f'Best Hyperparameters: {best_params}')
best_model = AlternatingLeastSquares(factors=best_params['factors'], regularization=best_params['regularization'], iterations=best_params['iterations'], use_gpu=True)
best_model.fit(train)
# Evaluate the model
precision = precision_at_k(best_model, train, test, K=5)
auc= AUC_at_k(best_model, train, test, K=5)
map = mean_average_precision_at_k(best_model, train, test, K=5)
ndcg = ndcg_at_k(best_model, train, test, K=5)


print(f'Precision@5: {precision:.4f}')
print(f'AUC@5: {auc:.4f}')
print(f'MAP@5: {map:.4f}')
print(f'NDCG@5: {ndcg:.4f}')


user_id = 17850  
n_recommendations = 10

user_train = train[user_id]
recommended_items = best_model.recommend(user_id, user_train, N=n_recommendations)

recommended_item_ids = [item[0] for item in recommended_items]

recommended_item_names = df[df['StockCodeEncoded'].isin(recommended_item_ids)]['Description']

print(f'User {user_id} might like:')
for item_name in recommended_item_names:
    print(item_name)


