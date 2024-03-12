# IMPORTANDO AS BIBLIOTECAS NECESSARIAS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# LENDO OS DADOS DAS SIMULAÇÕES ARMAZENADOS NUM ARQUIVO CSV

dados = pd.read_csv("/home/rafael/Python_scripts/Machine_learning/generated_database.csv", names=["volume_fraction", "field_intensity", "field_frequency", 
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

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size=0.3, random_state=126)

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


nn_model = MLPRegressor(hidden_layer_sizes=(500,), max_iter=1000, alpha=1e-4,solver='sgd', verbose=10)
nn_model.fit(train_scaled, y_train)

nn_model2 = MLPRegressor(hidden_layer_sizes=(500,), max_iter=1000, alpha=1e-4,solver='sgd', verbose=10)
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
#plt.scatter(previsaorf, alvo, s=10, c='green')
#plt.scatter(previsaodt, alvo, s=10, c='red')
#plt.scatter(previsaokn, alvo, s=10, c='pink')
plt.scatter(previsaonn, alvo, s=10, c='blue')
plt.plot(alvo, alvo)
plt.show()

#plt.xlim(250,550)
#plt.ylim(250,550)
#plt.scatter(previsaolr2, alvo2, s=10, c='black')
#plt.scatter(previsaorf2, alvo2, s=10, c='green')
#plt.scatter(previsaodt2, alvo2, s=10, c='red')
#plt.scatter(previsaokn2, alvo2, s=10, c='pink')
plt.scatter(previsaonn2, alvo2, s=10, c='blue')
plt.plot(alvo2, alvo2)
plt.show()

# CRIANDO AGORA DATAFRAMES DO PANDA COM NOVOS DADOS DE VARREDURAS PARA CHECAR O
# COMPORTAMENTO DAS PREVISÕES DOS MODELOS QUANDO FIXAMOS 3 DOS 4 INPUTS E EXECUTAMOS
# UMA VARREDURA NO QUARTO PARÂMETRO. FAREMOS ISSO PARA AS SEGUINTES VARIÁVEIS:
    
#   1 - FRAÇÃO VOLUMÉTRICA DE PARTÍCULAS
#   2 - CAMPO APLICADO
#   3 - FREQUÊNCIA DO CAMPO
#   4 - RAIO DAS NANOPARTÍCULAS

phi_min= 0.03
phi_max= 0.10
H_min = 2000
H_max = 10000
w_min = 100000
w_max = 350000
a_min = 5e-9
a_max = 1e-8

range_phi = np.linspace(phi_min,phi_max,1000)
range_H = np.linspace(H_min,H_max,1000)
range_w = np.linspace(w_min,w_max,1000)
range_a = np.linspace(a_min,a_max,1000)

phi_ref = 0.033
H_ref = 3000
w_ref = 184000
a_ref = 5e-9

varredura_phi = pd.DataFrame({"volume_fraction": range_phi, "field_intensity": H_ref, "field_frequency": w_ref, "particle_radius" : a_ref})
varredura_H = pd.DataFrame({"volume_fraction": phi_ref, "field_intensity": range_H, "field_frequency": w_ref, "particle_radius" : a_ref})
varredura_w = pd.DataFrame({"volume_fraction": phi_ref, "field_intensity": H_ref, "field_frequency": range_w, "particle_radius" : a_ref})
varredura_a = pd.DataFrame({"volume_fraction": phi_ref, "field_intensity": H_ref, "field_frequency": w_ref, "particle_radius" : range_a})

phi_scaled = scaler.transform(varredura_phi)
H_scaled = scaler.transform(varredura_H)
w_scaled = scaler.transform(varredura_w)
a_scaled = scaler.transform(varredura_a)

# PREVENDO O COMPORTAMENTO DA TEMPERATURA NO CENTRO DO TUMOR E DO TEMPO DE DESENVOLVIMENTO

# COM RELAÇÃO À FRAÇÃO VOLUMÉTRICA PHI

previsaolr_varredura_phi=lr_model.predict(phi_scaled) 
previsaolr2_varredura_phi=lr_model2.predict(phi_scaled) 

previsaorf_varredura_phi=rf_model.predict(phi_scaled) 
previsaorf2_varredura_phi=rf_model2.predict(phi_scaled) 

previsaodt_varredura_phi=tree_model.predict(phi_scaled) 
previsaodt2_varredura_phi=tree_model2.predict(phi_scaled) 

previsaokn_varredura_phi=kn_model.predict(phi_scaled) 
previsaokn2_varredura_phi=kn_model2.predict(phi_scaled) 

previsaonn_varredura_phi=nn_model.predict(phi_scaled) 
previsaonn2_varredura_phi=nn_model2.predict(phi_scaled) 


# COM RELAÇÃO À INTENSIDADE DO CAMPO H

previsaolr_varredura_H=lr_model.predict(H_scaled) 
previsaolr2_varredura_H=lr_model2.predict(H_scaled) 

previsaorf_varredura_H=rf_model.predict(H_scaled) 
previsaorf2_varredura_H=rf_model2.predict(H_scaled) 

previsaodt_varredura_H=tree_model.predict(H_scaled) 
previsaodt2_varredura_H=tree_model2.predict(H_scaled) 

previsaokn_varredura_H=kn_model.predict(H_scaled) 
previsaokn2_varredura_H=kn_model2.predict(H_scaled) 

previsaonn_varredura_H=nn_model.predict(H_scaled) 
previsaonn2_varredura_H=nn_model2.predict(H_scaled) 

# COM RELAÇÃO À FREQUENCIA DO CAMPO H

previsaolr_varredura_w=lr_model.predict(w_scaled) 
previsaolr2_varredura_w=lr_model2.predict(w_scaled) 

previsaorf_varredura_w=rf_model.predict(w_scaled) 
previsaorf2_varredura_w=rf_model2.predict(w_scaled) 

previsaodt_varredura_w=tree_model.predict(w_scaled) 
previsaodt2_varredura_w=tree_model2.predict(w_scaled) 

previsaokn_varredura_w=kn_model.predict(w_scaled) 
previsaokn2_varredura_w=kn_model2.predict(w_scaled) 

previsaonn_varredura_w=nn_model.predict(w_scaled) 
previsaonn2_varredura_w=nn_model2.predict(w_scaled) 


# COM RELAÇÃO AO RAIO DAS NANOPARTÍCULAS

previsaolr_varredura_a=lr_model.predict(a_scaled) 
previsaolr2_varredura_a=lr_model2.predict(a_scaled) 

previsaorf_varredura_a=rf_model.predict(a_scaled) 
previsaorf2_varredura_a=rf_model2.predict(a_scaled) 

previsaodt_varredura_a=tree_model.predict(a_scaled) 
previsaodt2_varredura_a=tree_model2.predict(a_scaled) 

previsaokn_varredura_a=kn_model.predict(a_scaled) 
previsaokn2_varredura_a=kn_model2.predict(a_scaled) 

previsaonn_varredura_a=nn_model.predict(a_scaled) 
previsaonn2_varredura_a=nn_model2.predict(a_scaled) 


# LENDO OS DADOS DAS VARREDURAS REALIZADAS POR MEIO DO SOFTWARE MHT2D

# VARREDURA DE PHI

dados_phi = pd.read_csv("/home/rafael/Python_scripts/Machine_learning/var_phi_ref.csv", names=["volume_fraction", "field_intensity", "field_frequency", "particle_radius", "tumour_temperature","time"], sep=" ")
dados_H = pd.read_csv("/home/rafael/Python_scripts/Machine_learning/var_H_ref.csv", names=["volume_fraction", "field_intensity", "field_frequency", "particle_radius", "tumour_temperature","time"], sep=" ")
dados_w = pd.read_csv("/home/rafael/Python_scripts/Machine_learning/var_w_ref.csv", names=["volume_fraction", "field_intensity", "field_frequency", "particle_radius", "tumour_temperature","time"], sep=" ")
dados_a = pd.read_csv("/home/rafael/Python_scripts/Machine_learning/var_a_ref.csv", names=["volume_fraction", "field_intensity", "field_frequency", "particle_radius", "tumour_temperature","time"], sep=" ")

X_phi = dados_phi["volume_fraction"]
T_phi = dados_phi["tumour_temperature"]
time_phi = dados_phi["time"]

X_H = dados_H["field_intensity"]
T_H = dados_H["tumour_temperature"]
time_H = dados_H["time"]

X_w = dados_w["field_frequency"]
T_w = dados_w["tumour_temperature"]
time_w = dados_w["time"]

X_a = dados_a["particle_radius"]
T_a = dados_a["tumour_temperature"]
time_a = dados_a["time"]


# ORGANIZANDO ESSAS COMPARAÇÕES NA FORMA DE GRÁFICOS

# PREVISÃO DA TEMPERATURA NO CENTRO DO TUMOR

figure, axis = plt.subplots(2,2, figsize=(10,8))
axis[0,0].plot(range_phi,previsaolr_varredura_phi)
#axis[0,0].plot(range_phi,previsaorf_varredura_phi)
axis[0,0].plot(range_phi,previsaonn_varredura_phi)
axis[0,0].scatter(X_phi,T_phi, s=10, c='black')
axis[0,0].set_ylim(35,60)
axis[0,0].set_title("Volume fraction")
axis[0,1].plot(range_H,previsaolr_varredura_H)
#axis[0,1].plot(range_H,previsaorf_varredura_H)
axis[0,1].plot(range_H,previsaonn_varredura_H)
axis[0,1].scatter(X_H,T_H, s=10, c='black')
axis[0,1].set_ylim(35,80)
axis[0,1].set_title("Field intensity")
axis[1,0].plot(range_w,previsaolr_varredura_w)
#axis[1,0].plot(range_w,previsaorf_varredura_w)
axis[1,0].plot(range_w,previsaonn_varredura_w)
axis[1,0].scatter(X_w,T_w, s=10, c='black')
axis[1,0].set_ylim(35,60)
axis[1,0].set_title("Field frequency")
axis[1,1].plot(range_a,previsaolr_varredura_a)
#axis[1,1].plot(range_a,previsaorf_varredura_a)
axis[1,1].plot(range_a,previsaonn_varredura_a)
axis[1,1].scatter(X_a,T_a, s=10, c='black')
axis[1,1].set_ylim(35,65)
axis[1,1].set_title("Particle radius")
plt.show

# PREVISÃO DO TEMPO PARA ATINGIR ESSA TEMPERATURA

figure, axis = plt.subplots(2,2, figsize=(10,8))
axis[0,0].plot(range_phi,previsaolr2_varredura_phi)
#axis[0,0].plot(range_phi,previsaorf2_varredura_phi)
axis[0,0].plot(range_phi,previsaonn2_varredura_phi)
axis[0,0].scatter(X_phi,time_phi, s=10, c='black')
#axis[0,0].set_ylim(35,70)
axis[0,0].set_title("Volume fraction")
axis[0,1].plot(range_H,previsaolr2_varredura_H)
#axis[0,1].plot(range_H,previsaorf2_varredura_H)
axis[0,1].plot(range_H,previsaonn2_varredura_H)
axis[0,1].scatter(X_H,time_H, s=10, c='black')
#axis[0,1].set_ylim(35,70)
axis[0,1].set_title("Field intensity")
axis[1,0].plot(range_w,previsaolr2_varredura_w)
#axis[1,0].plot(range_w,previsaorf2_varredura_w)
axis[1,0].plot(range_w,previsaonn2_varredura_w)
axis[1,0].scatter(X_w,time_w, s=10, c='black')
#axis[1,0].set_ylim(35,60)
axis[1,0].set_title("Field frequency")
axis[1,1].plot(range_a,previsaolr2_varredura_a)
#axis[1,1].plot(range_a,previsaorf2_varredura_a)
axis[1,1].plot(range_a,previsaonn2_varredura_a)
axis[1,1].scatter(X_a,time_a, s=10, c='black')
#axis[1,1].set_ylim(35,65)
axis[1,1].set_title("Particle radius")
plt.show