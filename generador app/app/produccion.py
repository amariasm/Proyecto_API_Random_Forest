from logging import FileHandler
import pickle
from pyexpat import model
import pandas as pd

url = 'https://raw.githubusercontent.com/anderfernandez/datasets/main/Casas%20Madrid/houses_Madrid.csv'
data = pd.read_csv(url )
data.info()

import numpy as np

str_cols = data.select_dtypes(['object']).columns
str_unique_vals = data[str_cols]\
    .apply(lambda x: len(x.dropna().unique()))

print(str_unique_vals)

print(data['has_garden'].unique())
print(data['has_pool'].unique())

str_unique_vals_cols = str_unique_vals[str_unique_vals == 1].index.tolist()

data.loc[:,str_unique_vals_cols] = data\
  .loc[:,str_unique_vals_cols].fillna(False)

 # Elimino variables con mucho NA
ind_keep = data.isna().sum() < 0.3 * data.shape[0]
data = data.loc[:,ind_keep]

# Remove columns
data.drop([
  'title', 'street_name','raw_address',
  'is_exact_address_hidden','is_rent_price_known',
  'is_buy_price_known', 'subtitle',
  'floor','buy_price_by_area', 'rent_price', 'id', 'Unnamed: 0'
  ], axis = 1, inplace = True)

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

# Cambio el tamaño
from matplotlib.pyplot import figure
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

str_cols = data.select_dtypes('object').columns.tolist()
num_cols = data.select_dtypes(['int', 'float']).columns.tolist()

# Selecciono datos numéricos
cor_matrix = pd.concat([data[num_cols]], axis = 1).corr()
sns.heatmap(cor_matrix)
plt.show()




# Cambio el tamaño
from matplotlib.pyplot import figure
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

str_cols = data.select_dtypes('object').columns.tolist()
num_cols = data.select_dtypes(['int', 'float']).columns.tolist()


from sklearn.model_selection import train_test_split

keep_cols = ['sq_mt_built', 'n_bathrooms', 'n_rooms' , 'has_lift', 'house_type_id']

# Split de los datos
y = data['buy_price']
x = data[keep_cols]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1234)

print(x_train.shape, y_train.shape)

x_train.isna().sum()
x_train\
  .assign(
      n_nas = x_train['has_lift'].isnull(),
      n_rows = 1
      )\
  .groupby('house_type_id')\
  .sum()\
  .reset_index()\
  .loc[:,['house_type_id', 'n_nas', 'n_rows']]


# Transformo en train y test
x_train.loc[
            x_train['house_type_id'] == 'HouseType 2: Casa o chalet', 'has_lift'
            ] = False

x_test.loc[
            x_test['house_type_id'] == 'HouseType 2: Casa o chalet', 'has_lift'
            ] = False

x_train.isna().sum()

# Imputo NAs con la moda

import pickle
# Calculo las modas
modes = dict(zip(x_train.columns, x_train.mode().loc[0,:].tolist()))

# Imputo la moda
for column in x_train.columns:
  x_train.loc[x_train[column].isna(),column] = modes.get(column)



from sklearn.metrics import mean_absolute_error  
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelBinarizer

# Defino el encoder
encoder = LabelBinarizer()
encoder_fit = encoder.fit(x_train['house_type_id'])

encoded_data_train = pd.DataFrame(
  encoder_fit.transform(x_train['house_type_id']),
  columns = encoder_fit.classes_.tolist()
) 

# Add encoded variables
x_train_transf = pd.concat(
  [x_train.reset_index(), encoded_data_train],
  axis = 1
  ).drop(['index', 'house_type_id'], axis = 1)

# Create model
rf_reg = RandomForestRegressor()
rf_reg_fit = rf_reg.fit(x_train_transf, y_train)

preds = rf_reg_fit.predict(x_train_transf)


mean_absolute_error(y_train, preds)
print("Error absoluto medio de train: ", mean_absolute_error(y_train, preds))

# Imputo la moda
for column in x_test.columns:
  x_test.loc[x_test[column].isna(),column] = modes.get(column)

# One hot encoding
encoded_data_test = pd.DataFrame(
  encoder_fit.transform(x_test['house_type_id']),
  columns = encoder_fit.classes_.tolist()
) 

x_test_transf = pd.concat(
  [x_test.reset_index(), encoded_data_test],
  axis = 1
  )\
  .drop(['index','house_type_id'], axis = 1)

preds = rf_reg_fit.predict(x_test_transf)

mean_absolute_error(y_test, preds)
print("Error absoluto medio de test: ", mean_absolute_error(y_test, preds))

 
 
with open('generador app/app/encoder.pickle', 'wb') as handle:
    pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('generador app/app/model.pickle', 'wb') as handle:
    pickle.dump(rf_reg, handle)