# IMPORTANDO AS BIBLIOTECAS NECESSARIAS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# LENDO OS DADOS DAS SIMULAÇÕES ARMAZENADOS NUM ARQUIVO CSV

dados = pd.read_csv("/home/rafael/generated_database7.csv", names=["volume_fraction", "field_intensity", "field_frequency", 
                                                           "particle_radius", "tumour_temperature","time"], sep=" ")


# IMPORTANDO DA BIBLIOTECA sklearn A FUNÇÃO PARA DIVIDIR OS DADOS QUE SERÃO
# USADOS PARA TREINAMENTO DO ALGORITMO DE MACHINE LEARNING ESCOLHIDO

from sklearn.model_selection import train_test_split

# DEFININDO OS DADOS DE INPUT A PARTIR DOS DADOS CONTIDOS NO ARQUIVO CSV
# AQUI OS DADOS DE INPUT SÃO TODAS AS COLUNAS COM EXCEÇÃO DAS DUAS VINCULADAS
# AOS OUTPUTS. NESSE PONTO DEFINIMOS INPUTS E OUTPUTS.

X = dados.drop(["tumour_temperature","time"], axis=1)
y = dados["tumour_temperature"]
y2 = dados["time"]

# AQUI USAMOS A FUNÇÃO train_test_split PARA DIVIDIR PARTE DOS DADOS DE INPUT
# EM DADOS QUE SERÃO USADOS PARA TREINAMENTO DO ALGORITMO. NESSE CASO A VARIÁVEL
# test_size DETERMINA O PERCENTUAL DE DADOS QUE NÃO SERÃO UTILIZADOS PARA TREINAR
# O ALGORITMO E QUE SERÃO UTILIZADOS APENAS PARA REALIZAÇÃO DE TESTES DE VERIFICAÇÃO.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=125)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size=0.3, random_state=125)

# AQUI FAZEMOS UMA NORMALIZAÇÃO DOS DADOS UTILIZADOS NO TREINAMENTO DO ALGORITMO
# DE MACHINE LEARNING. ISSO É FEITO TANTO PARA DOS INPUTS DE TREINAMENTO QUANTO
# PARA OS INPUTS VINCULADOS AOS TESTES DE PERFORMANCE.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(X_test)

train_scaled2 = scaler.fit_transform(X_train2)
test_scaled2 = scaler.transform(X_test2)

# AQUI IMPORTAMOS OS ALGORITMOS ESPECÍFICOS DE MACHINE LEARNING TESTADOS

from sklearn.tree import DecisionTreeRegressor #ALGORITMO DECISION TREE
from sklearn.ensemble import RandomForestRegressor # ALGORITMO RANDOM FORESTS
from sklearn.linear_model import LinearRegression # ALGORITMO DE REGRESSÃO LINEAR
from sklearn.neighbors import KNeighborsRegressor # ALGORITMO K-NEAREST NEIGHBORS
from sklearn.neural_network import MLPRegressor # ALGORITMO DE REDES NEURAIS


# AQUI ACIONAMOS ESSES ALGORITMOS

tree_model = DecisionTreeRegressor()
tree_model.fit(train_scaled, y_train)

tree_model2 = DecisionTreeRegressor()
tree_model2.fit(train_scaled2, y_train2)

rf_model = RandomForestRegressor()
rf_model.fit(train_scaled, y_train)

rf_model2 = RandomForestRegressor()
rf_model2.fit(train_scaled2, y_train2)


nn_model = MLPRegressor()
nn_model.fit(train_scaled, y_train)

nn_model2 = MLPRegressor()
nn_model2.fit(train_scaled2, y_train2)

lr_model = LinearRegression()
lr_model.fit(train_scaled, y_train)

lr_model2 = LinearRegression()
lr_model2.fit(train_scaled2, y_train2)

kn_model = KNeighborsRegressor()
kn_model.fit(train_scaled, y_train)

kn_model2 = KNeighborsRegressor()
kn_model2.fit(train_scaled2, y_train2)

# FAZENDO UMA AVALIAÇÃO DO ERRO ABSOLUTO NAS PREVISÕES DESSES ALGORITMOS

from sklearn.metrics import mean_absolute_error

tree_test_mae = mean_absolute_error(y_test, tree_model.predict(test_scaled))
rf_test_mae = mean_absolute_error(y_test, rf_model.predict(test_scaled))
lr_test_mae = mean_absolute_error(y_test, lr_model.predict(test_scaled))
kn_test_mae = mean_absolute_error(y_test, kn_model.predict(test_scaled))
nn_test_mae = mean_absolute_error(y_test, nn_model.predict(test_scaled))


tree_test_mae2 = mean_absolute_error(y_test2, tree_model2.predict(test_scaled2))
rf_test_mae2 = mean_absolute_error(y_test2, rf_model2.predict(test_scaled2))
lr_test_mae2 = mean_absolute_error(y_test2, lr_model2.predict(test_scaled2))
kn_test_mae2 = mean_absolute_error(y_test2, kn_model2.predict(test_scaled2))
nn_test_mae2 = mean_absolute_error(y_test2, nn_model2.predict(test_scaled2))


# IMPRIMINDO NO CONSOLE DE EXECUÇÃO DO SCRIPT ESSES ERROS


print("Algorithm errors for temperature prediction")

print("Linear regression mean absolute error = ",lr_test_mae)
print("Decision Tree test mean absolute error = ",tree_test_mae)
print("Random Forest test mean absolute error = ",rf_test_mae)
print("K Nearest Neighbors test mean absolute error = ",kn_test_mae)
print("Neural Networks test mean absolute error = ",nn_test_mae)

print("Algorithm errors for development time prediction")

print("Linear regression mean absolute error = ",lr_test_mae2)
print("Decision Tree test mean absolute error = ",tree_test_mae2)
print("Random Forest test mean absolute error = ",rf_test_mae2)
print("K Nearest Neighbors test mean absolute error = ",kn_test_mae2)
print("Neural Networks test mean absolute error = ",nn_test_mae2)


# A PREVISÃO DOS ALGORITMOS É DADA PELA CHAMADA ..._model.predict(test_scaled)

# PREVISÃO DO ALGORITMO DE REGRESSÃO LINEAR

previsaolr=lr_model.predict(test_scaled) 
previsaolr2=lr_model2.predict(test_scaled2) 


# PREVISÃO DO ALGORITMO RANDOM FORESTS

previsaorf=rf_model.predict(test_scaled) 
previsaorf2=rf_model2.predict(test_scaled2) 

# PREVISÃO DO ALGORITMO DECISION TREES

previsaodt=tree_model.predict(test_scaled) 
previsaodt2=tree_model2.predict(test_scaled2) 


# PREVISÃO DO ALGORITMO K NEAREST NEIGHBORS

previsaokn=kn_model.predict(test_scaled) 
previsaokn2=kn_model2.predict(test_scaled2) 

# PREVISÃO DO ALGORITMO DE REDES NEURAIS

previsaonn=nn_model.predict(test_scaled) 
previsaonn2=nn_model2.predict(test_scaled2) 

# O ALVO SÃO OS VALORES DA VARIÁVEL y_test COM EXCEÇÃO DA PRIMEIRA LINHA (INDEXADOR)

alvo=np.array(y_test)[:] 
alvo2=np.array(y_test2)[:] 


# PLOTANDO UMA COMPARAÇÃO DO ALVO (LINHA RETA) COM AS PREVISÕES (SÍMBOLOS) DOS
# OUTPUTS CORRESPONDENTES VINCULADOS A CADA ALGORITMO

#plt.xlim(35,70)
#plt.ylim(35,70)
#plt.scatter(previsaolr, alvo, s=10, c='black')
plt.scatter(previsaorf, alvo, s=10, c='green')
#plt.scatter(previsaodt, alvo, s=10, c='red')
#plt.scatter(previsaokn, alvo, s=10, c='pink')
#plt.scatter(previsaonn, alvo, s=10, c='blue')
plt.plot(alvo, alvo)
plt.show()

#plt.xlim(250,550)
#plt.ylim(250,550)
#plt.scatter(previsaolr2, alvo2, s=10, c='black')
plt.scatter(previsaorf2, alvo2, s=10, c='green')
#plt.scatter(previsaodt2, alvo2, s=10, c='red')
#plt.scatter(previsaokn2, alvo2, s=10, c='pink')
#plt.scatter(previsaonn2, alvo2, s=10, c='blue')
plt.plot(alvo2, alvo2)
plt.show()