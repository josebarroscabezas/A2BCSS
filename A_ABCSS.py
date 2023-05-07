# -*- coding: utf-8 -*-
"""
Created on Mon Jun 04 15:52:59 2018

@author: User
"""

import numpy as np
from tqdm import tqdm
from scipy import stats
from matplotlib import rc
rc('text', usetex=True)
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#%%
def size_muestra(p,e,c,N):
    z = stats.norm.ppf(1-(1-c)/2)
    n0 = z**2 * p * (1-p) / e**2
    n = n0/(1+(n0-1)/N)
    nk = n/N
    return nk

#Weighting functions 
def f_alpha(alpha):
    muw1 = 0.4
    stdw1 = 0.1
    
    fval = np.exp(-(alpha-muw1)**2/(2*stdw1**2))
    
    return fval

def f_tol2(g, p_0, p_0i, epsilon_0, epsilon_f):
    epsiloni = []
    N = len(g)
    nt = int(np.round(N*p_0))
    
    nti = np.round(N*p_0i)
    for i in range(np.size(nti)):               
        auxdist = g[int(nti[i]),-1]
        epsiloni.append(auxdist)
        ###AQUI. ES LA EVALUACIÓN CON HX DE THETA 1 Y 2 PARA g_ordenada[0:int(nti[i]),2]
            
    maxepsilon = np.max(epsiloni)
    minepsilon = np.min(epsiloni)
    
    epsilon = g[nt,-1]
    
    if maxepsilon == minepsilon:
        f = 1
    else:
        f = 1.0 - (epsilon - minepsilon)/(maxepsilon - minepsilon)
        
    return f

#%%
# ABC-SubSim
def ABCSS(g_ordenada, sigma, p_0, n_param, L, U, N, gRe, indices_parap0, g_parap0,otros,metrica):
    nt = int(np.round(N*p_0))
    ns = int(np.ceil(1/p_0 -1))
    contador = 1
    conta_elec = 0
    g_i = g_ordenada
    longi = nt + 1
    conta_parap0 = 0
    for i in range(nt):
        semilla = g_i[i, :]
        for j in range(ns):
            contador = contador + 1
            u = np.zeros(n_param)
            for k in range(n_param):
                done = 0
                while done == 0:
                    
                    u[k] = semilla[k] + np.random.randn()*sigma[k]/10 # MCR: the prop std has been reduced to increase the acc_rate
                    if u[k] >= L[k] and u[k] <= U[k]:
                        done = 1

            # Para ahorrar evaluaciones de la metrica
            if i in indices_parap0 and j == 0:
                g_temp = g_parap0[conta_parap0]
                contador = contador - 1
                conta_parap0 = conta_parap0 + 1
            else:
                g_temp = metrica(u,otros) # evaluación de función objetivo (F_obj). Métrica
            
            if g_temp <= g_ordenada[nt, n_param]:
                g_i[longi, :] = [*u, g_temp] # modificar según F_obj
                conta_elec = conta_elec + 1
            else:
                g_i[longi, :] = g_i[i, :]
            
            semilla = g_i[longi, range(n_param)]
            longi = longi + 1
            if longi >= N:
                break
       
        if longi >= N:
            break
        
    g_ordenada = g_i[g_i[:, n_param].argsort()] 
    
    sigma = np.max([np.std(g_ordenada[:, list(range(n_param))],0),(np.array(U)-np.array(L))/1000],0)
    gRe = np.append(gRe,np.array(g_ordenada[0,:], ndmin = 2), axis=0)
    max_sigma = np.max(sigma)
    toleval = g_ordenada[nt, n_param] #MCR: new tolerance evaluated in the newly produced subset
    
    ratio_elec = float(conta_elec) / contador
   
    return g_ordenada, max_sigma, sigma, ratio_elec, gRe, contador, toleval, conta_parap0

# P0 selection function
def parap0(g_i, N, p_0, n_param, sigma,L,U,nn,otros,metrica):
    contador = 0
    conta_elec = 0
    nt_i = int(np.floor(N*p_0))
    nt_ip0 = int(np.round(N*nn))
    ns = int(1)
    rho = g_i[nt_i, n_param]
    indices = np.random.choice(nt_i, nt_ip0, replace = False)
    indices = indices[indices[:].argsort()]
    g_i = g_i[indices]
    g_temp = np.zeros(N)
    for i in range(nt_ip0):
        semilla = g_i[i, :]
        for j in range(ns):
            contador = contador + 1
            u = np.zeros(n_param)
            for k in range(n_param):
                done = 0
                while done == 0:
                    u[k] = semilla[k] + np.random.randn()*sigma[k]/10
                    if u[k] >= L[k] and u[k] <= U[k]:
                        done = 1
            g_temp[contador - 1] = metrica(u,otros) # evaluación de función objetivo (F_obj)
            if g_temp[contador - 1] <= rho:
                semilla = g_temp[contador - 1]
                conta_elec = conta_elec + 1
    alpha = float(conta_elec) / contador

    w1 = f_alpha(alpha)

    return alpha, w1, contador, g_temp, indices

#%%
# p0 selection
def func(x,a,b,c):
    return -np.abs(a)*x**2 + b*x +c 

def devfunc(a,b,c):
    return b/(2*a)

def std_p0(p0,w):
    lista_p0 = []
    for i in range(len(p0)):
        if np.isnan(w[i]):
            num = 1
        else:
            num = int(np.ceil(1/np.abs((1.01-w[i]))))
        for j in range(num):
            lista_p0.append(p0[i])
    med = np.mean(lista_p0)
    desv = np.std(lista_p0)
    return med, desv

def diff_p0(p0):
    indices = [0]
    for i in range(1,len(p0)):
        for j in range(i):
            if np.abs(p0[j]-p0[i])<0.005:
                break
        if j == i-1:
            indices.append(i)
    para10 = []
    for i in range(len(indices),len(p0)):
        para10.append(0)
    return p0[indices],para10
                

# pruebo nn veces el algoritmo con p0 constante y con p0 variable para estimar
# cuántos pasos son necesarios para obtener la respuesta con la tolerancia
# adoptada. En "aa" voy grabando el número de subsets requeridos en cada prueba

#A2BCSS
def prueba(N,p_0,n_param,maxi_try,tolerance,L,U,otros,metrica):
    # nn = 0.02 # nn*N es el numero de evaluaciones utilizadas para definir p_0
#    t0 = time()
    ##############################################################################
    # Solución del problema
    ##############################################################################
    # Definición de parámetros de inicio
    # N = 5000 # número de datos
    # p_0 = 0.5 # probabilidad condicional
    p_0std = 0.75*p_0 # 0.25 #MCR: This is best for checking new values
    # n_param = 2 # número de parámetros
    # maxi_try = 20 # máximos pasos SubSets
    # tolerance = 80 # tolerancia para terminar SubSets
    # L = list([0.0001 , 0.01]) # límite inferior de los parámetros
    # U = list([0.02 , 2]) # límite superior de los parámetros
    nn = size_muestra(0.5,0.1,0.95,N)
    # Monte-Carlo inicial
    # print('Monte-carlo')
    g = np.zeros(shape = (N, n_param + 1))
    for i in range(n_param):
        g[:, i] = np.random.uniform(L[i],U[i],N)
    # print('metrica')
    print('Monte-Carlo:')
    for i in tqdm(range(N)):
        g[i, n_param] = metrica(g[i,:-1],otros)
    
    # Ordenar de menor a mayor los valores de la función objetivo
    g_ordenada = g[g[:,n_param].argsort()] # función objetivo
    gRe = np.array(g_ordenada[0, :], ndmin = 2) # voy grabando los mejores valores
    sigma = np.std(g_ordenada[:, list(range(n_param))],axis=0)
    
    epsilon_0 = g_ordenada[ N-1 , n_param]# MCR: Initial value for the epsilon taken from the Monte-Carlo

    ##############################################################################
    # ABC Sub_Sim
    ##############################################################################
    r_elec = np.zeros(maxi_try)
    
    ACC_Interm_SS = np.zeros(shape = (maxi_try + 1 , N , n_param + 1 ))
    ACC_Interm_SS[0 , : ]  = g_ordenada
    n_pruebas = 3
    w_acc = np.zeros((maxi_try, n_pruebas , 3))
    alpha_acc = np.zeros((maxi_try, n_pruebas , 1))#MCR: Inicia la matriz de acumulación de ratio de aceptación
    alpha_used_acc = np.zeros((maxi_try,  1))#MCR: Inicia la matriz de acumulación de ratio de aceptación usado en cada subset
    p0i_acc = np.zeros((maxi_try, n_pruebas , 1))#MCR: Inicia la matriz de acumulación de p0s
    p_0_used_acc = np.zeros((maxi_try,  1))#MCR: Inicia la matriz de acumulación de P_0 usado en cada subset
    ww_used_acc = []#np.zeros((maxi_try,  1))#MCR: Inicia la matriz de acumulación de W usado en cada subset
    cuenta_eval = 0
    ahorro = 0
    p_0_std = []
    
    print('A2BCSS:')
    for i in tqdm(range(maxi_try)):
        
        p_0_std.append(p_0std)
        #   ################################################
        # Revisiones para distintos p_0
        alpha = np.zeros(n_pruebas) # relaciones de aceptación para p_0 0.1, 0.2 y 0.5
        w1 = np.zeros(n_pruebas) # pesos en función a la relación de aceptación
        w2 = np.zeros(n_pruebas) # pesos en función de la tolerancia alcanzada
        indices = np.zeros((n_pruebas,int(np.round(N*nn)),1))
        if p_0std >= 0.001:
            p_0i = stats.truncnorm.rvs(a=(0.1 - p_0)/p_0std, b=(0.5-p_0)/p_0std,
                                   loc = p_0, scale = p_0std, size = n_pruebas)
        else:
            p_0i = np.array([p_0,0.1,0.5])
                
    #        p_0i,resto = diff_p0(p_0i)
    
        p0i_acc[i,:,0] = p_0i.tolist()
        g_temp = np.zeros((len(p_0i),N,1))
        
        for jj in range((len(p_0i))):
            n_evaluaciones = 0
            if p_0i[jj] != 0:
                alpha[jj], w1[jj], n_evaluaciones, g_temp[jj,:,0], indices[jj,:,0] = parap0(g_ordenada, N, p_0i[jj], n_param, sigma,L,U,nn,otros,metrica)
                w2[jj] = f_tol2(g_ordenada, p_0i[jj] , p_0i , epsilon_0, tolerance)
            cuenta_eval = cuenta_eval + n_evaluaciones

        # graba w1, w2 y w1*w2
        w_acc[i,:,0] = w1 
        w_acc[i,:,1] = w2
        w_acc[i,:,2] = w1*w2
        
        alpha_acc[i,:,0] = alpha #MCR: Accumula el ratio de aceptación
        ii = np.argmax(w1*w2)
        
        try:
            popt,pcov = curve_fit(func,p_0i,w1*w2)
            optim_p0 = devfunc(*popt)
        except:
            optim_p0 = 0.6
            pass
        
        if optim_p0 > 0.5 or optim_p0 < 0.1:
            p_0 = p_0i[ii]
        else:
            p_0 = optim_p0
        
        alpha_used = alpha[ii] #MCR: Obtiene el valor del r. aceptación finalmente usado
        alpha_used_acc[i] = alpha_used
        p_0_used_acc[i] = p_0
        indices_parap0 = indices[ii,:,0]
        g_parap0 = g_temp[ii,:,0]
        
        # sss = sigma
        g_ordenada, maxsigma, sigma, r_elec[i], gRe, contador, toleval, c = ABCSS(g_ordenada, sigma, p_0, n_param, L, U, N, gRe, indices_parap0, g_parap0,otros,metrica)
        cuenta_eval = cuenta_eval + contador
        ahorro = ahorro + c

        if toleval <= tolerance:
            break
        # if all(sss[i]==sigma[i] for i in range(len(sigma))):
        #     break
        ACC_Interm_SS[i + 1, : ]  = g_ordenada
        p_0,p_0std = std_p0(p_0i,w1*w2)
    ww_used_acc.append(w_acc)
    ACC_Interm_SS[i + 1, : ]  = g_ordenada

    # return i, cuenta_eval, alpha_used_acc, p_0_used_acc, ww_used_acc, p0i_acc,ACC_Interm_SS, ahorro,p_0_std
    return i, ACC_Interm_SS

def metrica(teta, otros):
    P_medida, d_medida, I, Luz = otros
    d_calculada = P_medida*(Luz*teta[1])**3/(48*teta[0]*I)
    rho = np.sum(np.abs(d_medida-d_calculada)/d_medida)
    return rho


if __name__ == "__main__":
    # valores medidos 
    P_medida= np.array([1000003.1, 1002718.4,  985205.2,
       1002402.7, 1008734.3, 1004342.3,  997693.7, 1017899.9, 1010384.9,
       1023816.8, 1012753.7,  982949.1, 1003451.8,  985190.0,  991086.0,
        978144.7,  981907.1,  997235.7,  986055.5,  992130.2])# np.random.normal(1000000,10000,20)
    d_medida = np.array([100.,  95.,  95.,  95.,
        95., 101., 100.,  92., 102., 100.,  96., 101.,
        93.,  98.,  95., 101.,  91., 105.,  98., 102.])# np.random.normal(100,5,20)
    I, Luz = 130208333.33, 5000.0
    N = 2000 # número de datos
    p_0 = 0.5 # probabilidad condicional
    p_0std = 0.75*p_0 # 0.25 #MCR: This is best for checking new values
    n_param = 2 # número de parámetros
    maxi_try = 5 # máximos pasos SubSets
    tolerance = 0.72 # tolerancia para terminar SubSets
    L = list([150000,0.98]) # límite inferior de los parámetros
    U = list([250000,1.02]) # límite superior de los parámetros
    otros = [P_medida, d_medida, I, Luz]
    a = prueba(N,p_0,n_param,maxi_try,tolerance,L,U,otros,metrica)
    
    plt.figure()
    plt.title('Elastic modulus estimation')
    plt.hist(a[1][0][:,0],label='Prior',density=True)
    plt.hist(a[1][a[0]+1][:,0],label='Posterior',density=True)
    plt.legend()
    
    


