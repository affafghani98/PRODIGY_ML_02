import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

train_data = pd.read_csv('train.csv')
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']#selecting features
target = 'SalePrice'#target variable
X = train_data[features]
y = train_data[target]
# Visualize relationship between features and target variable before training
fig, axes = plt.subplots(nrows=1, ncols=len(features), figsize=(15, 5))
fig.suptitle('Relationship between Features and Target Variable Before Training', y=1.05)
for i, feature in enumerate(features):
    axes[i].scatter(X[feature], y, alpha=0.5)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel(target)
plt.tight_layout()
plt.savefig('before_training.png') # Saving figure as PNG
plt.close()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)#splitting the data
imputer = SimpleImputer(strategy='mean')#handling missing values , if there
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
#Training my model
model = LinearRegression()
model.fit(X_train_imputed, y_train)
y_train_pred = model.predict(X_train_imputed)#predictions
y_val_pred = model.predict(X_val_imputed)
# Visualize relationship between features and target variable after training
fig, axes = plt.subplots(nrows=1, ncols=len(features), figsize=(15, 5))
fig.suptitle('Relationship between Features and Target Variable (After Training)', y=1.05)
for i, feature in enumerate(features):
    axes[i].scatter(X_train[feature], y_train, alpha=0.5, label='Actual')
    axes[i].scatter(X_train[feature], y_train_pred, color='red', alpha=0.5, label='Predicted')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel(target)
    axes[i].legend()
plt.tight_layout()
plt.savefig('after_training.png')  #Save figure as PNG
plt.close()
