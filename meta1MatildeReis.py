#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Introdução à Engenharia e Ciência de Dados
# --------------------------
# Matilde Martins Reis, nº 2021237887, (DEI/FCTUC)
# abril, 2022


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys


#---------------------------------------- 1. Aquisição de dados

C = np.loadtxt('COVID19.txt')
N = C.shape[0]  # 600 doentes ---> numero de linhas
M = C.shape[1] # numero de colunas 

#---------------Inquerito/Triagem
GN = C[:,0]  #genero
ID = C[:,1]  #idade
EC = C[:,2]  #estado civil
V  = C[:,3]  #vacinacao
DR = C[:,4]  #dificuldade em respirar

#---------------Diapositivos/sensores
RC = C[:,5]  #ritmo cardiaco
PA = C[:,6]  #pressao arterial
TP = C[:,7]  #temperatura

#--------------Decisão do médico
IT = C[:,8]  #INTERNAMENTO ou ALTA


# correlaçao
X = C[:,0:8] # entradas desejadas 
Y = C[:,8] # saídas desejadas
dias = range(0,N)

#---------------------------------------- 2. Pré-Processamento de dados

#-------------------------- 2.1 Valores em falta 

#substituir o valor -1 da pressao arterial -----> pela media 

idFalta = np.where(PA==-1)   # valores invalidos -- 15 
idValido = np.where(PA!=-1)  # valores validos 

PAValido = PA[idValido]
media = PA.mean()
mediaInt = media.round() # media da pressao arterial dos doentes = 137
PA[idFalta] = mediaInt  


print('\n::::::::: Valores em falta ----------', PA)
print()


#-------------------------- 2.2 Outliers
#substituir valores do ritmo cardiaco ---> utilizei o metodo de substituicao pela media nos valores q ultrapassam um certo limite fixo

limite = 100   # valor fixo


fora = np.where(RC > limite) # se o valor do ritmo cardiaco for sup ao lim --> outliers
fora = fora[0]

dentro = np.where (RC <= limite)
dentro = dentro[0]


mediaRC = RC.mean()
RCcopy = RC.copy()
RCcopy [fora] = mediaRC

print('Outliers',mediaRC)
print(fora)
print()

#GRAFICO
plt.figure(1)
plt.plot(dias, RC ,'ro',RCcopy,'bo')
plt.title('Outliers')
plt.xlabel('Pacientes')
plt.ylabel('Ritmo Cardiaco')
plt.grid()
plt.show()

#---------------------------------------- 3. TRANSFORMAÇAO DE DADOS

#-------------------------- 3.1 Seleção de variáveis 
# Optei por utilizar o Método de Filter (depende das carcteristicas das variaveis) onde utilizei os valores de correlação

# valores de correlaçao linear ou de pearson entre duas variaveis X e Y 

# entradas desejadas 

# col 0 genero
# col 1 idade  <---
# col 2 estado civil
# col 3 vacinacao  <---
# col 4 dificuldade respirar <---
# col 5 ritmo cardiaco
# col 6 pressao arterial
# col 7 temperatura <---


M = X.shape[1] # numero de colunas
tam = C.size  # 600 * 9 = 5400 , tamanho da matriz 
dias = range(0,N)

VC = np.zeros(M) # valores de correlaçao
for i in range(0,M):
    c = np.corrcoef(X[:,i],Y)
    ci = c[0,1]
    VC[i] = ci 
    
    
# escolher o valor maximo em valor absoluto
VC = np.abs(VC)    # valor absoluto 
escolher = np.argsort(VC) # ordenar os indices 
escolher = escolher[-4:] # escolhe as 4 variaveis finais as q teem maior correlação

print('\n:::::::::::: Metodo 2 - Uso de Correlação----------')
print('Valores de correlação',VC.round(4))
print('Variáveis Selecionadas:', escolher,'Correlaçao',VC[escolher].round(4))
print()

# escolhi a idade, vacinação, dificuldade em respirar e temperatura

# ignorei o genero, pois na minha opinião é irrelevante para um medico decidir se um paciente tem covid, entao eu ignoro e como a correlação a difculdade em respirar 
#é semelhante escolhi a dificuldade em respirar


#---------------------- visualizar saida desejada e entrada mais correlacionada
plt.figure(3)
plt.title('Valores de Correlação')
plt.plot(dias,Y,'r',dias,X[:,escolher],'b')
plt.legend(['Output', 'Input'])
plt.grid()


#-------------------------- 3.2 Resumir os dados

idAlta = np.where(IT == 0) # tem alta
idInternado = np.where(IT == 1) # fica internado


#-------media ---> idade
mediaIDA = ID[idAlta].mean().round() # media da idade dos doentes com alta
mediaIDI = ID[idInternado].mean().round() # media da idade dos doentes internados


print("Media da ID dos pacientes com alta:",mediaIDA)
print("Media da ID dos pacientes internados:",mediaIDI)
print()

#-------media ---> vacinação 
mediaVA = V[idAlta].mean().round() # media dos doentes vacinados com alta
mediaVI = V[idInternado].mean().round() # media dos doentes vacinados internados

print("Media dos pacientes VC com alta:",mediaVA)
print("Media os pacientes VC internados:",mediaVI)
print()

#-------media ---> dificuldade em respirar
mediaDRA = DR[idAlta].mean().round() # media dos doentes com dif em respirar com alta
mediaDRI = DR[idInternado].mean().round() # media dos doentes com dif em respirar internados

print("Media dos pacientes com DR com alta:",mediaDRA)
print("Media dos pacientes com DR internados:",mediaDRI)
print()

#-------media ---> temperatura
mediaTPA = TP[idAlta].mean().round() #media da temperatura dos doentes com alta
mediaTPI = TP[idInternado].mean().round() #media da temperatura dos doentes internados

print("Media da TP dos pacientes com alta:",mediaTPA)
print("Media da TP dos pacientes internados:",mediaTPI)
print()


mediaIdade     = 0.5*mediaIDA + 0.5*mediaIDI
mediaVacinaçao = 0.5*mediaVA + 0.5*mediaVI
mediaRespirar  = 0.5*mediaDRA + 0.5*mediaDRI
mediaTemperatura = 0.5*mediaTPA + 0.5*mediaTPI


print("Media ID:",mediaIdade)
print("Media VC:",mediaVacinaçao)
print("Media DR:",mediaRespirar)
print("Media TP:",mediaTemperatura)
print()


# desvio padrao para cada variavel 

desvioID = ID.std().round()
desvioV = VC.std().round()
desvioDR = DR.std().round()
desvioTP = TP.std().round()

print("Desvio padrao - Idade: ",desvioID)
print("Desvio padrao - Vacinação: ",desvioV)
print("Desvio padrao - Dif em Respirar: ",desvioDR)
print("Desvio padrao - Temperatura",desvioTP)
print()


#GRAFICO 
plt.figure(4)
plt.subplot(2,1,1)
plt.plot(ID[idAlta],V[idAlta],'ro',ID[idInternado],V[idInternado],'bo')
plt.legend(["Tem Alta","Internado"],loc="lower right")
plt.xlabel('Idade')
plt.ylabel('Vacinados')

plt.figure(5)
plt.subplot(2,2,4)
plt.plot(ID[idAlta],DR[idAlta],'ro',ID[idInternado],DR[idInternado],'bo')
plt.legend(["Tem Alta","Internado"],loc="lower right")
plt.xlabel('Idade')
plt.ylabel('Dificuldade Respirar')


#-------------------------- 4.1 Modelo de Classificação

#------------ 1.modelo regressivo

# desenvolver os minimos quadrados   (Xt*X)^-1 * XT*y 

X1 = C [:,[1,3,4,7]]         # matriz das variaveis escolhidas

X1T = np.transpose(X1)       # transposta 
X1TX = np.dot(X1T,X1)        # produto de duas matrizes 
pInv = np.linalg.inv(X1TX)   # inversa de matrizes
M1 = np.dot(pInv,X1T)        # produto de duas matrizes 
PAR = np.dot(M1,Y)           # produto de duas matrizes  


Ye2 = np.dot(X1,PAR).round() # produto das variveis escolhidas c o parametro

        
#------------ 4.2 Avaliacao usando SE = sensibilidade e SP = especificidade           

# avalidacao - quantificar a qualidade do modelo de classificacao    

TT = 0
FP = 0
TN = 0
FN = 0

for i in range(0,N):
    if Y[i] == Ye2[i] and Y[i] ==1:
        TT += 1
    if Y[i] == Ye2[i] and Y[i] == 0:  
        TN += 1
    if Y[i] == 1 and Ye2[i] == 0:  
        FN += 1
    if Y[i] == 0 and Ye2[i] == 1:
        FP += 1
        
SE = TT/(TT+FN)
SP = TN/(TN+FP) 

Recall = SE
Precision = TT/(TT+FP)

Fscore = 2 * (Precision * Recall) / (Precision + Recall)

print("\n::::::::: MODELO REGRESSIVO ----------------")
print("SE - Sensibilidade >",round(SE,3))
print("SP - Especificidade > " ,round(SP,3))  
print("Fscore ",round(Fscore,3))


#------------ 2.metodo KNN

NK = 3  # numero de vizinhos mais próximos
varEscolhidas = C[:, [1, 3, 4, 7]]  # matriz --> utilizamos todas as linhas e colunas 1,3,4,7
YT = np.zeros(N)  # YT ---> Vector de distancias, matriz c zeros -- 600 lin

for i in range(0, N):
    
    Dist = np.zeros(N)
    Pontoi = varEscolhidas[i,:] # ponto i --- corresponde a linha i onde o ciclo percorre todas as linhas
    
    for j in range(0,N):
        
        Pontoj = varEscolhidas[j,:]
        Dist[j] = np.linalg.norm(Pontoi - Pontoj) # faz a norma entre o ponto i e j 
    
    indSort = np.argsort(Dist) # ordena os indices por ordem crescente
    classes = IT[indSort[0:NK]]
    classesi = classes.mean().round() # calcula a media
    YT[i] = classesi 


#------------ 4.2 Avaliacao usando SE = sensibilidade e SP = especificidade        

# avalidacao - quantificar a qualidade do modelo de classificacao       

TT = 0
FP = 0
TN = 0
FN = 0

for i in range(0,N):
    if Y[i] == YT[i] and Y[i] ==1:
        TT += 1
    if Y[i] == YT[i] and Y[i] == 0:  
        TN += 1
    if Y[i] == 1 and YT[i] == 0:  
        FN += 1
    if Y[i] == 0 and YT[i] == 1:
        FP += 1

        
SE = TT/(TT+FN)
SP = TN/(TN+FP) 

Recall = SE
Precision = TT/(TT+FP)

Fscore = 2 * (Precision * Recall) / (Precision + Recall)

print("\n::::::::: METODO KNN ----------------")
print("SE - Sensibilidade >",round(SE,3))
print("SP - Especificidade > " ,round(SP,3))  
print("Fscore ",round(Fscore,3))


#------------ 3.modelo baseado em regras individuais

mediaIdade     = 0.5*mediaIDA + 0.5*mediaIDI
mediaVacinaçao = 0.5*mediaVA + 0.5*mediaVI
mediaRespirar  = 0.5*mediaDRA + 0.5*mediaDRI
mediaTemperatura = 0.5*mediaTPA + 0.5*mediaTPI

# criar um modelo  ----> decide se fica internado ou se tem alta

ITidade = np.zeros(N)
for i in range(0,N):
    if ID[i] > mediaIdade:
        ITidade[i] = 1    # internado 
  

ITvacinaçao = np.zeros(N)
for i in range(0,N):
    if V[i] > mediaVacinaçao:
        ITvacinaçao[i] = 1   # internado 
        
        
ITrespirar = np.zeros(N)
for i in range(0,N):
    if DR[i] > mediaRespirar:
        ITrespirar[i] = 1   # internado 
        
ITtemperatura = np.zeros(N)
for i in range(0,N):
    if TP[i] > mediaTemperatura:
        ITtemperatura[i] = 1   # internado 



Ye3 = ((ITidade+ITvacinaçao+ITrespirar+ITtemperatura)/4).round()


#------------ 4.2 Avaliacao usando SE = sensibilidade e SP = especificidade  

# avalidacao - quantificar a qualidade do modelo de classificacao           

TT = 0
FP = 0
TN = 0
FN = 0

for i in range(0,N):
    if Y[i] == Ye3[i] and Y[i] ==1:
        TT += 1
    if Y[i] == Ye3[i] and Y[i] == 0:  
        TN += 1
    if Y[i] == 1 and Ye3[i] == 0:  
        FN += 1
    if Y[i] == 0 and Ye3[i] == 1:
        FP += 1
        
SE = TT/(TT+FN)
SP = TN/(TN+FP) 

Recall = SE
Precision = TT/(TT+FP)

Fscore = 2 * (Precision * Recall) / (Precision + Recall)

print("\n::::::::: MODELO BASEADO EM REGRAS INDIVIDUAIS ----------------")
print("SE - Sensibilidade >",round(SE,3))
print("SP - Especificidade > " ,round(SP,3))
print("Fscore ",round(Fscore,3))