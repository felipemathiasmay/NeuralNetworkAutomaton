import numpy as np
import warnings as wr
from sklearn import neural_network as nn
from sklearn.exceptions import ConvergenceWarning


#     Sujeito             Simbologia      Númerico          
'''
      Galinha                 G              00
 Viagem fazendeiro            F     (Estado "irrelevante")
  Volta fazendeiro            V              01
     Cachorro                 C              10
      Alface                  A              11

          Trabalhando com a solução GVAAAGGGCVG
'''

# taxa de aprendizado
lr = 0.03

#dataset de treino
#pos.automato q0 q1 q2 q3 q4 q5 q6 q7 q8 q9 q10 Simb.
X = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],      #Simb = 00 (G)
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],      #Simb = 01 (V)
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],      #Simb = 11 (A)
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],      #Simb = 11 (A)
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],      #Simb = 11 (A)
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],      #Simb = 00 (G)
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],      #Simb = 00 (G)
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],      #Simb = 00 (G)
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],      #Simb = 10 (C)
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],      #Simb = 01 (V)
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])     #Simb = 00 (G)

# rotulos do dataset de treino
T = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],      
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],      
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],      
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],      
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],      
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],      
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],     
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],      
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],      
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],      
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])     

# dataset de testes
#pos.automato q0 q1 q2 q3 q4 q5 q6 q7 q8 q9 q10 Simb.
Z = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],      #Simb = 00 (G)
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],      #Simb = 01 (V)
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],      #Simb = 11 (A)
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],      #Simb = 11 (A)
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],      #Simb = 11 (A)
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],      #Simb = 00 (G)
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],      #Simb = 00 (G)
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],      #Simb = 00 (G)
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],      #Simb = 10 (C)
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],      #Simb = 01 (V)
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])     #Simb = 00 (G)

mlp = nn.MLPClassifier(hidden_layer_sizes=(20,), max_iter=128, alpha=1e-4, solver='sgd', verbose=10, random_state=1, learning_rate_init=lr)

# treino
print('#################### EXECUCAO ####################')
print('Treinamento') 
with wr.catch_warnings():
    wr.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X, T)

# teste
print('Testes') 
Y = mlp.predict(Z)

# resultado 
print('Resultado procurado') 
print(T)
print("Score de treino: %f" % mlp.score(X, T))
print('Resultado encontrado') 
print(Y)
print("Score do teste: %f" % mlp.score(Z, T))

sumY = [sum(Y[i]) for i in range(np.shape(Y)[0])] # saida
sumT = [sum(T[i]) for i in range(np.shape(T)[0])] # target
