import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Preprocesado y modelado
# ==============================================================================
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf

'''Primero creamos un dataFrame con los datos'''
df = pd.DataFrame(columns=['Observacion', 'Tiempo de entrega', 'Cantidad de cajas', 'Distancia'])

mb= pd.read_csv("TER.csv")
list = []
dictionary={}
i = 0
for element in mb:
    list.append(element)
    #dictionary[observacion] =
    #dictionary[]
    #dictionary[]
    #dictionary[]
    i+=1
    #print(element)
j =0
count=1
w=[]
x=[]
y=[]
z=[]
for item in list:

    if j == 0:
        x.append(item)
        j+=1
    elif j == 1:
        y.append(item)
        j += 1
    else:
        z.append(item)
        w.append(count)
        count += 1
        j = 0

print(w)
print(x)
print(y)
print(z)

df.head()
'''División de los datos en train y test'''
def train():
    X_train, X_test, y_train, y_test = train_test_split(
        X.values.reshape(-1, 1),
        y.values.reshape(-1, 1),
        train_size=0.8,
        random_state=1234,
        shuffle=True
'''Creacion el modelo'''
def modelo():
    modelo = LinearRegression()
    modelo.fit(X=X_train.reshape(-1, 1), y=y_train)
    print("Intercept:", modelo.intercept_)
    print("Coeficiente:", list(zip(X.columns, modelo.coef_.flatten(), )))
    print("Coeficiente de determinación R^2:", modelo.score(X, y))