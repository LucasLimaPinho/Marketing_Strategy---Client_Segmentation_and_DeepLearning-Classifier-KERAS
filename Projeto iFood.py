"""
iFOOD - Data Science Challenge

Created on Thu May 14 07:06:12 2020

Python Version 3.7.0
Keras version 2.3.1
Model: Deeplearning Classifier (Binary Classification)
Main references:
    
    1. https://keras.io/ (Built on top of Tensorflow 2.0)
    2. https://www.coursera.org/specializations/deep-learning 
    Deeplearning Specialization ensinada por Andrew Ng - professor 
    de Ciência da Computação em Stanford

Sobre o autor:
    . Nome: Lucas Lima Reis de Pinho
    . Cargo Atual: Coordenador de Engenharia na HEINEKEN 
        com Foco em Transformação Digital. Entrei como Trainee em 2018
    . Mestrado em Otimização Computacional (COPPE-UFRJ) - maior CRA
    . MBA em Gerenciamento de Projetos (FGV)
    . FGV Executive Big Data & Machine Learning Program
    . Especialização em Deeplearning pela deeplearning.ai (Univ. Stanford)
    . LinkedIn: https://www.linkedin.com/in/lucaslimapinho/
    . Github: https://github.com/LucasLimaPinho/Data-Science
        

"""


from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ConfusionMatrix
from keras.models import Sequential
from keras.optimizers import Adam,SGD
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.preprocessing import LabelEncoder
import math
import statistics
import keras
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.metrics.pairwise import pairwise_distances_argmin
import datetime as dt



##### ANÁLISE EXPLORATÓRIA DE DADOS #####

# Loading data a partir do próprio desktop
# Inicialmente 28 colunas de atributos
# 1 class (Binary Classification -> aderiu a campanha (1) ou não (0)

# Observação_1: todos os dados da coluna Z_CostContact = 3 e não impactam na classificação
# Observação_2: todos os dados da coluna Z_Revenue = 11 e não impactam na classificação
# Observação_3: a identificação do cliente (Coluna ID) não impactam na classificação
# Observação_4: o modelo KERAS necessita de Categorical Encoding: transformar
# variáveis categóricas em numéricas. Utilizaremos o método LabelEncoder()
# Observação_5: o atributo (Income) impacta diretamente a propensão de adesão as campanhas
# e apresenta alguns valores NaN. Estes serão excluídos do modelo para não impactar
# negativamente no treinamento da máquina

dados = pd.read_csv('C:/Users/Francisco/Desktop/Projeto_iFood/ml_project1_data.csv')
dados['Response'] = dados['Response'].astype(int)
dados = dados.dropna() # Apagando linhas com valores NaN para Income.
dados.insert(0,'year',pd.DatetimeIndex(dados['Dt_Customer']).year)
dados.insert(1,'month',pd.DatetimeIndex(dados['Dt_Customer']).month)
dados.insert(2,'day',pd.DatetimeIndex(dados['Dt_Customer']).day)                         
del dados['ID']
del dados['Z_CostContact']
del dados['Z_Revenue']
del dados['Dt_Customer']       
matrix_correlation = dados.corr() 


# Verificar as correlações entre os atributos.
# Reduzir o número de atributos com correlação elevada pode evitar o DUMMY VARIABLE TRAP 
# Segue abaixo o mapa de correlações entre os atributos
# A correlação ajuda a entender como um atributo se comporta em relação ao outro
# Correlação >= 0.7 (positiva e forte)
# Correlação entre 0.3 e .7: correlação positiva e média
# Correlação entre 0 e 0.3: correlação positiva e média
# Correlação 0 -> correlação nula entre atributos
# Correlação entre -0.3 e 0 -> correlação negativa e média
# Correlação entre -0.3 e -0.7 --> correlação negativa e média
# Correlação entre -0.7 e -1.0 --> correlação negativa e forte


sn.heatmap(matrix_correlation, annot=True)
plt.show() 

# A partir daqui podemos retirar diversos insights sobre o comportamento do cliente.

segmentacao_clientes = dados.corr().unstack().sort_values(ascending = False).drop_duplicates()
print("*------------------------------------------------*")
print("*------------------------------------------------*")
print("As cinco correlações positivas mais fortes entre atributos de clientes são: ")
print(segmentacao_clientes[1:5])
print("                                                   ")
print("                                                   ")
print("*------------------------------------------------*")
print("*------------------------------------------------*")
print("*------------------------------------------------*")
print("                                                   ")
print("                                                   ")
print("As cinco correlações negativas mais fortes entre atributos de clientes são: ")
print(segmentacao_clientes[-6:-1])

""" ALGUNS INSIGHTS: consumidores que gastam muito dinheiro com carnes
(MnTMeatProducts) também realizam muitas compras em catálogos.
Consumidores que consomem bastante peixe (MntFishPrducts) também consomem
frutas (MntFruits) """



# Outro ponto importante é enxergar algumas variáveis que podem ter mais impacto na
# resposta ser igual a 1. Isso pode ser feito adotando probabilidades condicionais.

# Podemos ver aqui que temos aproximadamente 85% de possibilidade de termos valor igual 0
# Temos aproximadamente 15% de chance de resposta ser igual 1 - conforme enunciado disse.
colunas = list(dados.columns)
rating_probs = dados.groupby('Response').size().div(len(dados)) 
print("Probabilidade de Resposta ser negativa =" + str(rating_probs[0]))
print("Probabilidade de Resposta ser positiva =" + str(rating_probs[1]))
#agrupando as possibilidades de ter resposta negativa para todos os valores de atributos
# Podemos analisar a probabilidade condicional para alguns atributos que intuitivamente
# impactam mais na possibilidade de termos uma resposta positiva como grau de escolaridade
# e renda. Seguem alguns exemplos:

Income_q1 = np.logical_and(dados['Income'] <= dados['Income'].quantile(0.25),
                           dados['Response'] == 1)
Probs_q1 = sum(Income_q1)/len(dados)
print("A probabilidade de a resposta à ativação de marketing seja postiva para indivíduos com renda inferior a R$:" + str(dados['Income'].quantile(0.25)) + " é igual a " + str(Probs_q1 * 100) + " %")

# Observe que temos apenas 15% dos dados com resposta positiva. Isto pode dificultar o treinamento
# do modelo de Deeplearning. Por isso, devemos realizar um trade_off de reduzir a quantidade de dados
# mas manter uma melhor distribuição entre respostas positiva e negativas para melhor treinamento.

#### REALIZANDO SEGMENTAÇÃO DE CLIENTES VIA RFM#############
# Recency: Quem tem comprado recentemente? Atributo: Recency
# Frequency: Quem tem comprado com frequência?  
# Monetary: Quem gasta mais dinheiro em compras?
# Quais atributos iremos focar?
# Outra forma de segmentação de clientes é utilizando algoritmos de 
# aprendizado não supervisionado de máquina como KMeans, KMedoids, AffinityPropagation

dados.head()
dados_RFM = pd.DataFrame() 
dados_RFM['Recency'] = dados['Recency']
col_list = list(dados[['NumDealsPurchases',
                       'NumCatalogPurchases',
                       'NumStorePurchases',
                       'NumWebPurchases']])
dados_RFM['Frequency'] = dados[col_list].sum(axis=1)
col_list2 = list(dados[['MntFishProducts',
                       'MntMeatProducts',
                       'MntFruits',
                       'MntSweetProducts',
                       'MntWines',
                       'MntGoldProds']])
dados_RFM['Monetary'] = dados[col_list].sum(axis=1)

dados_RFM['r_quartile'] = pd.qcut(dados_RFM['Recency'], 4, ['1','2','3','4'])
dados_RFM['f_quartile'] = pd.qcut(dados_RFM['Frequency'], 4, ['4','3','2','1'])
dados_RFM['m_quartile'] = pd.qcut(dados_RFM['Monetary'], 4, ['4','3','2','1'])
dados_RFM['RFM_Score'] = dados_RFM.r_quartile.astype(str)+ dados_RFM.f_quartile.astype(str) + dados_RFM.m_quartile.astype(str)

# Listando Clientes com maior SCORE RFM
top_clientes = dados_RFM[dados_RFM['RFM_Score']=='111'].sort_values('Monetary', ascending=False)
print("Os Customers_IDS dos melhores clientes são: ")
print("                                             ")
print("                                             ")
print(top_clientes.head)

dados_RFM['RFM_Score'] = dados_RFM['RFM_Score'].astype(int)
dados_RFM.groupby('RFM_Score')['Recency','Frequency','Monetary'].mean()
dados_RFM['Segment'] = 'Low-Value'
dados_RFM.loc[dados_RFM['RFM_Score']<400,'Segment'] = 'Mid-Value' 
dados_RFM.loc[dados_RFM['RFM_Score']<200,'Segment'] = 'High-Value'


##### CONSTRUINDO MODELO DE MACHINE LEARNING #####
    
"""Construindo os conjuntos de treinamento e teste. Para valores de samples
abaixo de 10.000, é possível utilizar a taxa 70% (treino) e 30% (teste). A 
partir do momento que subimos a quantidade de amostras (>10^6), 
aconselha-se moldar para algo mais próximo de 98% (train), 1 % (dev) e 1% (test)"""

X = dados.iloc[:,0:dados.shape[1]-1].values #Destacando os atributos da base de dados
Y = dados.iloc[:,dados.shape[1]-1].values # Destacando a classe da base de dados
#Categorical Encoding usando LabelEncoder para os atributos.
# Agora alguns atributos categóricos como nível de escolaridade (PhD, Mestrado, etc) serão substituídos
# atributos numéricos


labelencoder = LabelEncoder()
X[:,4] = labelencoder.fit_transform(X[:,4])
X[:,5] = labelencoder.fit_transform(X[:,5])  

""" Após preprocessamento e Análise Exploratória de Dados
    1) 27 atributos
    2) 1 classe (Binary Classification)
    3) Excluídas variáveis que não impactam e com alta correlação (DUMMY TRAP)
    4) Datas de Inclusão na base separadas em ano, mês e dia -> podem 
    oferecer informações importantes para o modelo quando segregadas
    5) Feito LABEL ENCODING para oferecer variáveis numéricas para o KERAS """"    
x_training, x_test, y_training, y_test = train_test_split(X,
                                                          Y,
                                                          test_size = 0.3,
                                                          random_state = 0)
# FEATURE SCALLING - Normalização com Z_Score
# Agiliza o processo de aprendizagem da máquina
# Todos valores centrados ao redor de média 0 e desvio padrão 1  
#norm = Normalizer()
#x_training=norm.fit_transform(x_training)
#x_test = norm.fit_transform(x_test)
sc = StandardScaler()
x_training = sc.fit_transform(x_training)
x_test = sc.fit_transform(x_test)

# Construindo o modelo machine learning para binary classification
# Possíveis funções de ativação:

# Z = w.T * X + b
# 1. Sigmoid Function --> a(z) = (1/(1+e^(-z)))
# 2. Rectified Linear Unit (ReLU) --> a(z) = max(0,z)
# 3. Leaky Rectified Linear Unit (Leaky ReLU) --> a(z) = max(0.01z,z), 
# the parameter 0.01 may change
# 4. Hyperbolic Tangent (tanh) --> a(z) = (e^z - e^-z / e^z + e^-z)
# 5. Softmax --> a(z) = e^z / sum(e^z): Softmax é importante quando precisamos
# o modelo de predição para mais de 2 classes.

# Para nosso caso de Binary Classification na última camada da rede neural,
# a função de ativação indicada para a última camada é a Sigmóide (Regressão Logística)

# Utilizando KERAS, adicionamos camadas com o número de hidden units (neurônios da camada)
# e explicitamos a função de ativação a ser utilizada. O termo DENSE refere-se a um neurônio
# com ligação densa - sinapses de entrada e saída com todos os neurônios de entrada e saída
# O termo Dropout pode ser utilizado como técnica de Regularização para reduzir overfitting
# de modelos. O Dropout refere-se a porcentagem de chance que de um neurônio ser desativado
# durante a propagação de uma sample.


modelo = Sequential()
modelo.add(Dense(units = 1024, kernel_initializer = 'uniform', 
                 activation ='relu',
           input_dim=27))


modelo.add(Dense(units = 512, kernel_initializer = 'uniform', 
                 activation ='relu'))

modelo.add(Dense(units = 256, kernel_initializer = 'uniform', 
                 activation ='relu'))

modelo.add(Dense(units = 256, kernel_initializer = 'uniform', 
                 activation ='relu'))
modelo.add(Dense(units = 256, kernel_initializer = 'uniform', 
                 activation ='relu'))
modelo.add(Dense(units = 128, kernel_initializer = 'uniform', 
                 activation ='relu'))
modelo.add(Dense(units = 128, kernel_initializer = 'uniform', 
                 activation ='relu'))

modelo.add(Dense(units = 64, kernel_initializer = 'uniform', 
                 activation ='relu'))

modelo.add(Dense(units = 64, kernel_initializer = 'uniform', 
                 activation ='relu'))

modelo.add(Dense(units = 64, kernel_initializer = 'uniform', 
                 activation ='relu'))
modelo.add(Dense(units = 64, kernel_initializer = 'uniform', 
                 activation ='relu'))
modelo.add(Dense(units = 16, kernel_initializer = 'uniform', 
                 activation ='relu'))
modelo.add(Dense(units = 16, kernel_initializer = 'uniform', 
                 activation ='relu'))
modelo.add(Dense(units = 16, kernel_initializer = 'uniform', 
                 activation ='relu'))
modelo.add(Dense(units = 16, kernel_initializer = 'uniform', 
                 activation ='relu'))
modelo.add(Dense(units = 16, kernel_initializer = 'uniform', 
                 activation ='relu'))

modelo.add(Dense(units = 16, kernel_initializer = 'uniform', 
                 activation ='relu'))

modelo.add(Dense(units = 16, kernel_initializer = 'uniform', 
                 activation ='relu'))

modelo.add(Dense(units = 1, kernel_initializer = 'uniform', 
                 activation ='sigmoid'))
modelo.summary()

# ALGUNS MÉTODOS DE OTIMIZAÇÃO DOS PARÂMETROS DA REDE NEURAL
# 1. Gradient Descent 
# 2. Gradient Descent with Momentum 
# 3. RMSProp - Root Mean Squared Propagation
# 4. ADAM (Adaptative Moment Estimation): 

#FORWARD PROPAGATION -> cálculo da função de custo a ser minimizada pelo 
# algoritmo de otimização

# BACKWARD PROPAGATION -> cálculo dos valores atuais da rede neural, 
# principalmente (weight, bias). 

# OTIMIZAÇÃO: Atualização dos parâmetros buscando reduzir a função de custo.
# Além de (weight, bias), outros parâmetros a serem atualizados são GAMA e BETA 
# para casos de utilização de Batch Normalization.



optimizer = Adam(learning_rate=0.001)
modelo.compile(optimizer = optimizer, loss = 'binary_crossentropy',
               metrics = ['accuracy'])


historico = modelo.fit(x_training, y_training, batch_size = 16, epochs = 1000, 
           validation_data = (x_test, y_test))
plt.plot(historico.history['loss'])
plt.plot(historico.history['accuracy'])

# Após montagem e ajuste do modelo, podemos medir a performance utilizando as 
# previsões com o conjunto de dados de teste (x_teste)
# Podemos utilizar as métricas para medir a performance do modelo de rede neural.

Y_pred = modelo.predict(x_test)
# Sigmoid como função de ativação da camada de saída nos dá resultados
# em termos de probabilidade. Devemos transformar probabilidades em
# actual classifications se temos a previsão que irá responder ou não
# as campanhas de marketing
Y_pred = (Y_pred > 0.5)


score = modelo.evaluate(x_test, y_test,verbose=1)

print(score)

# score é uma lista que possui o valor de 'loss' e 'accuracy'. 
# É bom lembrar que os dados disponíveis estão desbalanceados - muito mais exemplos
# de não resposta (0) do que resposta positiva às ações (1) no conjunto de teste.
# O "accuracy" muito bom pode está refletindo a distribuição da classe com muitas respostas negativas.


# A matriz de confusão ajuda a comparar as previsões em uma tabela mostrando as previsões
# corretas e os tipos de erros cometidos. Idealmente, a matriz de confusão é uma matriz diagonal
# Precisão será a medida da exatidão do classificador montado. Maior precisão, mais acurácia tem o classificador.
# Recall é uma medida da capacidade de cobertura do classificador. Quanto maior o recall, mais casos o classificador irá cobrir.
# Recall is a measure of a classifier’s completeness. The higher the recall, the more cases the classifier covers.
# F1 Score é média ponderada de precision e recall.

confusion = confusion_matrix(y_test, Y_pred)
print (confusion_matrix)
taxa_acerto = accuracy_score(y_test, Y_pred)
taxa_erro = 1 - taxa_acerto
print(taxa_acerto)

"""Com custo total da campanha em 6.720MU e taxa de 15% de acerto, tivemos
retorno de 3.674 MU e profit de - 3.046 MU. Fazendo uma regra de três básica
considerando que não há aumento dos custos de campanha (apenas um direcionamento
dado pelo modelo de Deeplearning), podemos inferir que o retorno com 88% de taxa
de acerto do modelo seria de 21554 MU e teríamos um profit de 14835 MU"""
