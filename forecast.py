import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error

df=pd.read_csv('./Cellphone.csv')

#  Матрица корреляций
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
plt.savefig('matrix.png')

# Удаление ненужных колонки
df = df.drop(["Product_id"], axis=1)
df = df.drop(["thickness"], axis=1)

# Много усов 1
nrows=6
ncols=2
fig,axes=plt.subplots(nrows=nrows,ncols=ncols,figsize=(15,25))
for i , col in enumerate(df.drop(['Price'],axis=1).columns):
    sns.boxplot(df[col],ax=axes[i//ncols,i%ncols])
plt.savefig('usi1.png')

# Обзор
df.head()
# Проверка пустых значений
df.isnull().sum()
# Проверка дублирующихся значений
df.duplicated().sum()

#  Матрица корреляций
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
plt.savefig('matrix.png')

# Обработка ppi
n1,n2=np.percentile(df['ppi'],[25,75])
iqr=n2-n1
lower_bound=n1-(1.5*iqr)
upper_bound=n2+(1.5*iqr)
plt.subplot(1,2,1)
plt.boxplot(df['ppi'])
plt.title('До обработки ppi')
df['ppi']= np.where(df['ppi']>upper_bound, upper_bound,df['ppi'])
plt.subplot(1,2,2)
plt.boxplot(df['ppi'])
plt.title('После обработки ppi')
plt.savefig('ppi.png')

# Обработка cpu
n1,n2=np.percentile(df['cpu freq'],[25,75])
iqr=n2-n1
lower_bound=n1-(1.5*iqr)
upper_bound=n2+(1.5*iqr)
plt.subplot(1,2,1)
plt.boxplot(df['weight'])
plt.title('До обработки CPU')
df['cpu freq']= np.where(df['cpu freq']< lower_bound,0.5,df['cpu freq'])
plt.subplot(1,2,2)
plt.boxplot(df['cpu freq'])
plt.title('После обработки CPU')
plt.savefig('CPU.png')

# weight
n1,n2=np.percentile(df['weight'],[25,75])
iqr=n2-n1
lower_bound=n1-(1.5*iqr)
upper_bound=n2+(1.5*iqr)
plt.subplot(1,2,1)
plt.boxplot(df['weight'])
plt.title('До обработки wight')
df['weight']= np.where(df['weight']>upper_bound, upper_bound,np.where(df['weight']< lower_bound,lower_bound,df['weight']))
plt.subplot(1,2,2)
plt.boxplot(df['weight'])
plt.title('После обработки wight')
plt.savefig('weight.png')

# resolution
n1,n2=np.percentile(df['resoloution'],[25,75])
iqr=n2-n1
lower_bound=n1-(1.5*iqr)
upper_bound=n2+(1.5*iqr)
plt.subplot(1,2,1)
plt.boxplot(df['resoloution'])
plt.title('До обработки resolution')
df['resoloution']= np.where(df['resoloution']>=10.1, 10.1,np.where(df['resoloution']< lower_bound,lower_bound,df['resoloution']))
plt.subplot(1,2,2)
plt.boxplot(df['resoloution'])
plt.title('После обработки resolution')
plt.savefig('resolution.png')

# Много усов 2
nrows=6
ncols=2
fig,axes=plt.subplots(nrows=nrows,ncols=ncols,figsize=(15,25))
for i , col in enumerate(df.drop(['Price'],axis=1).columns):
    sns.boxplot(df[col],ax=axes[i//ncols,i%ncols])
plt.savefig('usi2.png')

# Разделение на тестовую и обучающую выборки
y = df.Price
x = df.drop("Price", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=1)

# Нормализация
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(x_train)
x_train = pd.DataFrame(minmax_scaler.transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(minmax_scaler.transform(x_test), columns=x_train.columns)

# Стандартизация
standard_scaler = StandardScaler()
standard_scaler.fit(x_train)
x_train = pd.DataFrame(standard_scaler.transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(standard_scaler.transform(x_test), columns=x_train.columns)

# Преобразование данных в массивы Numpy
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Линейная регрессия
print("Линейная регрессия")
Model = LinearRegression()
Model.fit(x_train, y_train)
R2 = Model.score(x_train, y_train)
ypredict = Model.predict(x_test)
mape = mean_absolute_percentage_error(y_test, ypredict)*100
print('R2 =', R2)
print('MAPE =', mape, end="\n\n")

# Случайный лес
print("Случайный лес")
Model = RandomForestRegressor()
Model.fit(x_train,y_train)
R2 = Model.score(x_train, y_train)
ypredict=Model.predict(x_test)
mape = mean_absolute_percentage_error(y_test, ypredict)*100
print('R2 =', R2)
print('MAPE =', mape, end="\n\n")

# Предсказания случайного леса
res=pd.DataFrame({'Actual Values':y_test,
'Predicted Values':ypredict})
print(res)

# Метод ближайшего соседа
print("Метод ближайшего соседа")
Model = KNeighborsRegressor()
Model.fit(x_train, y_train)
ypredict = Model.predict(x_test)
R2 = Model.score(x_train, y_train)
mape = mean_absolute_percentage_error(y_test, ypredict)*100
print('R2 =', R2)
print('MAPE =', mape, end="\n\n")

# Дерево решений
print("Дерево решений")
Model = DecisionTreeRegressor()
Model.fit(x_train, y_train)
ypredict = Model.predict(x_test)
R2 = Model.score(x_train, y_train)
mape = mean_absolute_percentage_error(y_test, ypredict)*100
print('R2 =', R2)
print('MAPE =', mape, end="\n\n")
