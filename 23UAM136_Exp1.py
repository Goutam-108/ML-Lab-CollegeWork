import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('tested.csv')
print("Original Dataset :")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nDescription of dataset :")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

df.drop(columns = ['Cabin'], inplace = True)
imputer = SimpleImputer(strategy = 'mean')
df['Age'] = imputer.fit_transform(df[['Age']])
imputer = SimpleImputer(strategy = 'most_frequent')
df['Fare'] = imputer.fit_transform(df[['Fare']]).ravel()
print("\nAfter Handling Missing Data:")
print(df.isnull().sum())

categorical_cols = df.select_dtypes(include = 'object').columns.tolist()
print("\nCategorical Columns:", categorical_cols)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)
print("\nAfter Encoding Categorical Variables and Dropping Irrelevant Columns:")
print(df.head())

numerical_cols = df.select_dtypes(include = ['int64', 'float64']).columns.tolist()
print("\nNumerical Columns:", numerical_cols)
scaler = StandardScaler();
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
print("\nAfter Scaling (StandardScaler):")
print(df[numerical_cols].head())

Y = df['Survived']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("\nShapes of Train/Test Sets:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", Y_train.shape)
print("y_test:", Y_test.shape)
