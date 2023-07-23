import random
import numpy as np
# Choose random values for x1i, x2i, y1i, y2i, zi, β0i, β1i, β2i, β3i, and β4i




q = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71}

x1i = random.randint(1, len(q)-1)
x2i = random.randint(1, len(q)-1)
y1i = random.randint(1, len(q)-1)
y2i = random.randint(1, len(q)-1)
zi = random.randint(1, len(q)-1)
beta0i = random.randint(1, len(q)-1)
beta1i = random.randint(1, len(q)-1)
beta2i = random.randint(1, len(q)-1)
beta3i = random.randint(1, len(q)-1)
beta4i = random.randint(1, len(q)-1)
gamma1 = random.randint(1, len(q)-1)
gamma2 = random.randint(1, len(q)-1)
z = random.randint(1, len(q)-1)
print(f"q = {q}")
print(f"x1i = {x1i}")
print(f"x2i = {x2i}")
print(f"y1i = {y1i}")
print(f"y2i = {y2i}")
print(f"zi = {zi}")
print(f"beta0i = {beta0i}")
print(f"beta1i = {beta1i}")
print(f"beta2i = {beta2i}")
print(f"beta3i = {beta3i}")
print(f"beta4i = {beta4i}")
print(f"gamma1 = {gamma1}")
print(f"gamma2 = {gamma2}")
print(f"z = {z}")
p={2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
   113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199}

g1 = random.randint(1, len(p)-1)
g2 = random.randint(1,  len(p)-1)
print(f"g1 = {g1}")
print(f"g2 = {g2}")

h1 = pow(int(g1), int(gamma1))
print("La valeur de h1 est :", h1)

h2 = pow(int(g2), int(gamma2))
print("La valeur de h2 est :", h2)



# Compute private key sk
x1 = random.randint(1, 100)
x2 = random.randint(1, 100)
y1 = random.randint(1, 100)
y2 = random.randint(1, 100)
z = random.randint(1, 100)

sk = (x1, x2, y1, y2, z)

print(" sk =",sk)

# Calculer E1i, E2i, E3i, E4i et E5i
E1i = pow(int(g1), int(x1i)) * pow(int(h1), int(beta1i))
E2i = pow(int(g1), int(y1i)) * pow(int(h1), int(beta2i))
E3i = pow(int(g2), int(x2i)) * pow(int(h2), int(beta3i))
E4i = pow(int(g2), int(y2i)) * pow(int(h2), int(beta4i))
E5i = pow(int(g1), int(zi))
print("La valeur de E1i est :", E1i)
print("La valeur de E2i est :", E2i)
print("La valeur de E3i est :", E3i)
print("La valeur de E4i est :", E4i)
print("La valeur de E5i est :", E5i)

# Diffuser les valeurs E1i, E2i, E3i, E4i et E5i
broadcast_data = (E1i, E2i, E3i, E4i, E5i)
############################

import numpy as np

t = 5

# Génère aléatoirement les coefficients des polynômes
f_coef = np.random.randint(0, 10, size=t+1)
f_prime_coef = np.random.randint(0, 10, size=t)
g_coef = np.random.randint(0, 10, size=t+1)
g_prime_coef = np.random.randint(0, 10, size=t)
h_coef = np.random.randint(0, 10, size=t+1)
h_prime_coef = np.random.randint(0, 10, size=t)

# Crée des objets de polynômes à partir des coefficients générés
f = np.poly1d(f_coef)
f_prime = np.poly1d(f_prime_coef)
g = np.poly1d(g_coef)
g_prime = np.poly1d(g_prime_coef)
h = np.poly1d(h_coef)
h_prime = np.poly1d(h_prime_coef)

print("Le polynôme f(x) est :", f)
print("Le polynôme f'(x) est :", f_prime)
print("Le polynôme g(x) est :", g)
print("Le polynôme g'(x) est :", g_prime)
print("Le polynôme h(x) est :", h)
print("Le polynôme h'(x) est :", h_prime)

import numpy as np

t = 5

# Génère aléatoirement les coefficients des polynômes
a_coef = np.random.rand(t+1)*10
b_coef = np.random.rand(t)*10
a_prime_coef = np.random.rand(t+1)*10
b_prime_coef = np.random.rand(t)*10
a_prime_prime_coef = np.random.rand(t+1)*10
b_prime_prime_coef = np.random.rand(t)*10

# Crée des objets de polynômes à partir des coefficients générés
f = np.poly1d(a_coef)
f_prime = np.poly1d(b_coef)
g = np.poly1d(a_prime_coef)
g_prime = np.poly1d(b_prime_coef)
h = np.poly1d(a_prime_prime_coef)
h_prime = np.poly1d(b_prime_prime_coef)

print("Le polynôme f(x) est :", f)
print("Le polynôme f'(x) est :", f_prime)
print("Le polynôme g(x) est :", g)
print("Le polynôme g'(x) est :", g_prime)
print("Le polynôme h(x) est :", h)
print("Le polynôme h'(x) est :", h_prime)

###############"""

import numpy as np

t = 5
x1i = np.random.randint(-10, 10)

# Génère aléatoirement les coefficients des termes du polynôme
f_coef = np.random.rand(t+1)*10
f_coef[0] = x1i - sum(f_coef[1:]*(-1)**np.arange(1, t+1))

# Crée un objet de polynôme à partir des coefficients générés
f = np.poly1d(f_coef)

print("Le polynôme f(x) est :", f)
print("f(-1) =", f(-1))

#############
import numpy as np

t = 5
y1i = np.random.randint(-10, 10)

# Génère aléatoirement les coefficients des termes du polynôme
f_coef = np.random.rand(t+1)*10
f_coef[0] = y1i - sum(f_coef[1:]*(-2)**np.arange(1, t+1))

# Crée un objet de polynôme à partir des coefficients générés
f = np.poly1d(f_coef)

print("Le polynôme f(x) est :", f)
print("f(-2) =", f(-2))

##################
import numpy as np

t = 5
beta1i = np.random.randint(-10, 10)

# Génère aléatoirement les coefficients des termes du polynôme
f_prime = np.random.rand(t+1)*10
f_prime[0] = beta1i - sum(f_prime[1:]*(-1)**np.arange(1, t+1))

# Crée un objet de polynôme à partir des coefficients générés
f = np.poly1d(f_prime)

print("Le polynôme f(x) est :", f)
print("f'(-1) =", f(-1))
##################################
import numpy as np

t = 5
beta2i = np.random.randint(-10, 10)

# Génère aléatoirement les coefficients des termes du polynôme
f_prime = np.random.rand(t+1)*10
f_prime[0] = beta2i - sum(f_prime[1:]*(-2)**np.arange(1, t+1))

# Crée un objet de polynôme à partir des coefficients générés
f = np.poly1d(f_prime)

print("Le polynôme f(x) est :", f)
print("f'(-2) =", f(-2))
####################################""
import numpy as np

t = 5
x2i = np.random.randint(-10, 10)

# Génère aléatoirement les coefficients des termes du polynôme
g_coef = np.random.rand(t+1)*10
g_coef[0] = x2i - sum(g_coef[1:]*(-1)**np.arange(1, t+1))

# Crée un objet de polynôme à partir des coefficients générés
g = np.poly1d(g_coef)

print("Le polynôme g(x) est :", g)
print("g(-1) =", g(-1))
###########""""""""
import numpy as np

t = 5
y2i = np.random.randint(-10, 10)

# Génère aléatoirement les coefficients des termes du polynôme
g_coef = np.random.rand(t+1)*10
g_coef[0] = y2i - sum(g_coef[1:]*(-2)**np.arange(1, t+1))

# Crée un objet de polynôme à partir des coefficients générés
g = np.poly1d(g_coef)

print("Le polynôme g(x) est :", g)
print("g(-2) =", g(-2))

#######################
import numpy as np

t = 5
beta3i = np.random.randint(-10, 10)

# Génère aléatoirement les coefficients des termes du polynôme
g_prime = np.random.rand(t+1)*10
g_prime[0] = beta3i - sum(g_prime[1:]*(-2)**np.arange(1, t+1))

# Crée un objet de polynôme à partir des coefficients générés
g = np.poly1d(g_prime)

print("Le polynôme g(x) est :", g)
print("g'(-1) =", g(-1))

#########"
import numpy as np

t = 5
beta4i = np.random.randint(-10, 10)

# Génère aléatoirement les coefficients des termes du polynôme
g_prime = np.random.rand(t+1)*10
g_prime[0] = beta4i - sum(g_prime[1:]*(-2)**np.arange(1, t+1))

# Crée un objet de polynôme à partir des coefficients générés
g = np.poly1d(g_prime)

print("Le polynôme g(x) est :", g)
print("g'(-2) =", g(-2))
###########"""
import numpy as np

t = 5
zi= np.random.randint(-10, 10)
a_prime_prime_coef = np.random.rand(t+1)*10
# Génère aléatoirement les coefficients des termes du polynôme
h_coef = np.random.rand(t+1)*10

h_coef[0] =(zi  )

# Crée un objet de polynôme à partir des coefficients générés


print("Le polynôme h(x) est :", h_coef)

##########
import numpy as np

t = 5
beta0i= np.random.randint(-10, 10)
b_prime_prime_coef = np.random.rand(t)*10
# Génère aléatoirement les coefficients des termes du polynôme
h_prime = np.random.rand(t+1)*10
h_prime[0] =(beta0i)

# Crée un objet de polynôme à partir des coefficients générés


print("Le polynôme h'(x) est :", h_prime)

######################"





# Génère aléatoirement les coefficients des polynômes f(x), g(x) et h(x)
# Génère aléatoirement les coefficients des polynômes
a_coef = np.random.rand(t+1)*10
b_coef = np.random.rand(t)*10
a_prime_coef = np.random.rand(t+1)*10
b_prime_coef = np.random.rand(t)*10
a_prime_prime_coef = np.random.rand(t+1)*10
b_prime_prime_coef = np.random.rand(t)*10

# Crée des objets de polynômes à partir des coefficients générés
f = np.poly1d(a_coef)
f_prime = np.poly1d(b_coef)
g = np.poly1d(a_prime_coef)
g_prime = np.poly1d(b_prime_coef)
h = np.poly1d(a_prime_prime_coef)
h_prime = np.poly1d(b_prime_prime_coef)


f_coef = np.random.randint(0, 10, size=t+1)

g_coef = np.random.randint(0, 10, size=t+1)

h_coef = np.random.randint(0, 10, size=t+1)

# Calcule les termes de la somme CMik
t=3
for k in range(t+1):
    CMik = g_coef[k] * a_coef[k] * h_coef[k] * b_coef[k] * g_coef[k] * a_prime_coef[k] * h_coef[k] * b_prime_coef[k]
    print("La valeur de CMik est : ")
    print(CMik)

    cmik = g_coef[k] * a_prime_prime_coef[k] * h_coef[k] * b_prime_prime_coef[k]

    print("La valeur de cmik est : ")
    print(cmik)
    
   
    cmi0 = g_coef[k] * zi * h_coef[k] * beta0i
  
   # for i in range(1, n+1):
    #for tau in range(1, 3):
        #for j in range(1, n+1):
           # Yt = (tau + 2, i)
           # for k in range(tau+1):
               # E_tau_i = Yt * sum([CMik[-tau+k][i]])
                # print(E_tau_i)
                
                
Qtemp = ['participant1', 'participant2', 'participant3', 'participant4', 'participant5']  # Liste des participants qualifiés initiale
 

for i in Qtemp:
    Qtemp.remove(i)
    print("Participant Pi a été disqualifié. Nouvelle liste de participants qualifiés : ", Qtemp)
else:
    print("Le participanti n'est pas présent dans la liste des participants qualifiés.")
    
    
    n = 5
participants = ["P1", "P2", "P3"]
shares = {}

# Define some arbitrary functions
def fi(j):
    return j**2

def f_prime_i(j):
    return j**3

def gi(j):
    return 2*j

def g_prime_i(j):
    return 3*j

def hi(j):
    return j + 1

def h_prime_i(j):
    return j - 1

# Compute the shares for each participant and index
for i in range(1, n+1):
    for participant in participants:
        sfij = fi(i)
        sf_prime_ij = f_prime_i(i)
        sgij = gi(i)
        sg_prime_ij = g_prime_i(i)
        shij = hi(i)
        sh_prime_ij = h_prime_i(i)
        
        shares[(participant, i)] = (sfij, sf_prime_ij, sgij, sg_prime_ij, shij, sh_prime_ij)

# Print the shares for a specific participant and index
print(shares[("P2", 3)])
    
   


# Compute the left-hand side of the equation with extra indentation
#(pow(int(g1), int(sfij))*pow(int(h1), int(sf_prime_ij))* pow(int(g2), int(sgij))*pow(int(h2), int(sg_prime_ij)))= CMik * j**k

##################"etape4-a################""""""
import random

# Définir l'ensemble de valeurs possibles pour Sj
Sj = [100,25,63,58,77,91]

# Choisir des valeurs aléatoires pour aj, bj, a'j, b'j, a''j et b''j
aj = random.choice(Sj)
bj = random.choice(Sj)
a_prime_j = random.choice(Sj)
b_prime_j = random.choice(Sj)
a_double_prime_j = random.choice(Sj)
b_double_prime_j = random.choice(Sj)

# Créer l'ensemble S'j à partir des valeurs aléatoires choisies
S_prime_j = {aj, bj, a_prime_j, b_prime_j, a_double_prime_j, b_double_prime_j}
print("s'j :" , S_prime_j)

# Définir les constantes g1, g2, h1 et h2
g1 = 3
g2 = 5
h1 = 7
h2 = 11
pow(int(h1), int(b_double_prime_j))
# Calculer les valeurs de S''j à partir des valeurs aléatoires choisies
S_double_prime_j = {pow(int(g1), int(aj)), pow(int(g2), int(a_prime_j)), pow(int(g1), int(a_double_prime_j)), pow(int(h1), int(bj)), pow(int(h2), int(b_prime_j)), pow(int(h1), int(b_double_prime_j))}
print("s''j :" ,S_double_prime_j)

###################4-b####################
import random

# Définir l'ensemble de valeurs possibles pour Sk
Sk = [101,111,141,121,321,412]

# Choisir des valeurs aléatoires pour ak, bk, a'k, b'k, a''k et b''k
ak = random.choice(Sk)
bk = random.choice(Sk)
a_prime_k = random.choice(Sk)
b_prime_k = random.choice(Sk)
a_double_prime_k = random.choice(Sk)
b_double_prime_k = random.choice(Sk)

# Créer l'ensemble S'k à partir des valeurs aléatoires choisies
S_prime_k = {ak, bk, a_prime_k, b_prime_k, a_double_prime_k, b_double_prime_k}
print("s'k :" , S_prime_k)

# Définir les constantes g1, g2, h1 et h2
g1 = 3
g2 = 5
h1 = 7
h2 = 11

# Calculer les valeurs de S''k à partir des valeurs aléatoires choisies

S_double_prime_k = {pow(int(g1), int(ak)), pow(int(g2), int(a_prime_k)), pow(int(g1), int(a_double_prime_k)), pow(int(h1), int(bk)), pow(int(h2), int(b_prime_k)), pow(int(h1), int(b_double_prime_k))}
print("s''k :" , S_double_prime_k)

########################4-c################

# Définir l'ensemble Q de participants
Qtemp = ['P1', 'P2', 'P3', 'P4']

# Définir la valeur sfij
sfij = 2

# Définir la constante g1
g1 = 3

# Choisir une valeur aléatoire pour aj
aj = random.choice(Sj)

# Calculer le produit de toutes les valeurs g1 * ak pour chaque participant dans Q
product_g1_ak = 1
for Pk in Qtemp:
    ak = random.choice(Sk)
    product_g1_ak *= g1 * ak

# Calculer λ1 en utilisant la formule donnée
lambda_1 = sfij * g1 * aj * product_g1_ak
print ("λ1 : ", lambda_1)

# Définir l'ensemble Q de participants
Qtemp = ['P1', 'P2', 'P3']

# Définir la valeur sgij
sgij = 2

# Définir la constante g2
g2 = 5

# Choisir une valeur aléatoire pour a'j
a_prime_j = random.choice(Sj)

# Calculer le produit de toutes les valeurs g2 * a'k pour chaque participant dans Q
product_g2_aprime_k = 1
for Pk in Qtemp:
    a_prime_k = random.choice(Sk)
    product_g2_aprime_k *= g2 * a_prime_k

# Calculer λ'1 en utilisant la formule donnée
lambda_prime_1 = sgij * g2 * a_prime_j * product_g2_aprime_k
print ("λ'1 : ", lambda_prime_1)

# Définir l'ensemble Q de participants
Qtemp = ['P1', 'P2', 'P3']

# Définir la valeur shij
shij = 3

# Définir la constante g1
g1 = 7

# Choisir une valeur aléatoire pour a''j
a_double_prime_j = random.choice(Sj)

# Calculer le produit de toutes les valeurs g1 * a''k pour chaque participant dans Q
product_g1_adoubleprime_k = 1
for Pk in Qtemp:
    a_double_prime_k = random.choice(Sk)
    product_g1_adoubleprime_k *= g1 * a_double_prime_k

# Calculer λ''1 en utilisant la formule donnée
lambda_double_prime_1 = shij * g1 * a_double_prime_j * product_g1_adoubleprime_k
print ("λ''1 : ", lambda_double_prime_1)

# Définir la valeur sfij
sfij = 2

# Définir la constante g1
g1 = 3

# Choisir une valeur aléatoire pour aj
aj = random.choice(Sj)

# Calculer λ2 en utilisant la formule donnée
lambda_2 = sfij * (g1 * aj)
print ("λ2 : ", lambda_2)

# Définir la valeur sgij
sgij = 2

# Définir la constante g2
g2 = 5

# Choisir une valeur aléatoire pour a'j
a_prime_j = random.choice(Sj)

# Calculer λ'2 en utilisant la formule donnée
lambda_prime_2 = sgij * (g2 * a_prime_j)
print ("λ'2 : ", lambda_prime_2)

# Définir la valeur shij
shij = 3

# Définir la constante g1
g1 = 7

# Choisir une valeur aléatoire pour a''j
a_double_prime_j = random.choice(Sj)

# Calculer λ''2 en utilisant la formule donnée
lambda_double_prime_2 = shij * (g1 * a_double_prime_j)
print ("λ''2 : ", lambda_double_prime_2)

# Définir l'ensemble Q de participants
Q = ['P1', 'P2', 'P3', 'P4']

# Définir la valeur sf'ij
sf_prime_ij = 2

# Définir la constante h1
h1 = 5

# Choisir une valeur aléatoire pour bj
bj = random.choice(Sj)

# Calculer le produit de toutes les valeurs h1 * bk pour chaque participant dans Q
product_h1_bk = 1
for Pk in Q:
    b_k = random.choice(Sk)
    product_h1_bk *= h1 * b_k

# Calculer γ1 en utilisant la formule donnée
gamma_1 = sf_prime_ij * h1 * bj * product_h1_bk
print (" γ1 : ", gamma_1)

# Définir l'ensemble Q de participants
Q = ['P1', 'P2', 'P3', 'P4']

# Définir la valeur sg'ij
sg_prime_ij = 3

# Définir la constante h2
h2 = 7

# Choisir une valeur aléatoire pour b'j
b_prime_j = random.choice(Sj)

# Calculer le produit de toutes les valeurs h2 * b'k pour chaque participant dans Q
product_h2_b_prime_k = 1
for Pk in Q:
    b_prime_k = random.choice(Sk)
    product_h2_b_prime_k *= h2 * b_prime_k

# Calculer γ'1 en utilisant la formule donnée
gamma_prime_1 = sg_prime_ij * h2 * b_prime_j * product_h2_b_prime_k
print (" γ'1 : ", gamma_prime_1)

# Définir l'ensemble Q de participants
Q = ['P1', 'P2', 'P3', 'P4']

# Définir la valeur sh'ij
sh_prime_ij = 5

# Définir la constante h1
h1 = 3

# Choisir une valeur aléatoire pour b''j
b_double_prime_j = random.choice(Sj)

# Calculer le produit de toutes les valeurs h1 * b''k pour chaque participant dans Q
product_h1_b_double_prime_k = 1
for Pk in Q:
    b_double_prime_k = random.choice(Sk)
    product_h1_b_double_prime_k *= h1 * b_double_prime_k

# Calculer γ''1 en utilisant la formule donnée
gamma_double_prime_1 = sh_prime_ij * h1 * b_double_prime_j * product_h1_b_double_prime_k
print (" γ''1 : ", gamma_double_prime_1)

# Définir la valeur sf'ij
sf_prime_ij = 2

# Définir la constante h1
h1 = 5

# Choisir une valeur aléatoire pour bj
bj = random.choice(Sj)

# Calculer γ2 en utilisant la formule donnée
gamma_2 = sf_prime_ij * (h1 * bj)
print (" γ2 : ", gamma_2)

# Définir la valeur sg'ij
sg_prime_ij = 3

# Définir la constante h2
h2 = 7

# Choisir une valeur aléatoire pour b'j
b_prime_j = random.choice(Sj)

# Calculer γ'2 en utilisant la formule donnée
gamma_prime_2 = sg_prime_ij * (h2 * b_prime_j)
print (" γ'2 : ", gamma_prime_2)

# Définir la valeur sh'ij
sh_prime_ij = 5

# Définir la constante h1
h1 = 3

# Choisir une valeur aléatoire pour b''j
b_double_prime_j = random.choice(Sj)

# Calculer γ''2 en utilisant la formule donnée
gamma_double_prime_2 = sh_prime_ij * (h1 * b_double_prime_j)
print (" γ''2 : ", gamma_double_prime_2)
#################4-d############"


parts = [aj, bj, a_prime_j, b_prime_j, a_double_prime_j, b_double_prime_j] # Exemple de parts
parts = [ak, bk, a_prime_k, b_prime_k, a_double_prime_k, b_double_prime_k]

print(" les parts sont  envoyees ", parts)
#################4-e###############

# Calculer α en utilisant la formule donnée
alpha = lambda_1 / lambda_2
print('α = ', alpha)
alpha_prime = lambda_prime_1 / lambda_prime_2
print(" α' = " ,  alpha_prime)
alpha_double_prime = lambda_double_prime_1 / lambda_double_prime_2
print(" α'' = " ,  alpha_double_prime )
beta = gamma_1 / gamma_2
print(" β = " ,  beta )
beta_prime = gamma_prime_1 / gamma_prime_2
print(" β' = " ,  beta_prime )
beta_double_prime = gamma_double_prime_1 / gamma_double_prime_2
print(" β'' = " ,  beta_double_prime )
# Définir la séquence de valeurs
seq = [1,2,3,4,5]

# Calculer r en utilisant la formule donnée
r = sum(seq)
print('r=', r)
# Définir la séquence de valeurs
sequence_prime = [3, 6, 9, 12, 15]

# Calculer r' en utilisant la formule donnée
r_prime = sum(sequence_prime)
print("r'=", r_prime)

# Définir la séquence de valeurs
sequence_double_prime = [1, 3, 5, 7, 9]

# Calculer r'' en utilisant la formule donnée
r_double_prime = sum(sequence_double_prime)
print("r''=", r_double_prime)

# Définir la séquence de valeurs
sequence = [0.5, 1.2, 2.1, 3.3, 5.4]

# Calculer t en utilisant la formule donnée
t = sum(sequence)
print("t=", t)
# Définir la séquence de valeurs
sequence_prime = [1.2, 2.4, 3.6, 4.8, 6.0]

# Calculer t' en utilisant la formule donnée
t_prime = sum(sequence_prime)
print("t'=", t_prime)

# Définir la séquence de valeurs
sequence_double_prime = [0.1, 0.3, 0.5, 0.7, 0.9]

# Calculer t'' en utilisant la formule donnée
t_double_prime = sum(sequence_double_prime)
print("t''=", t_double_prime)



# Calculer α1r en utilisant la formule donnée
alpha_1_r = pow(int(g1), int(sfij))

print ('α/1r =' , alpha_1_r)

alpha_prime_1_r_prime = pow(int(g2), int(sgij))
print ("α'/1r'=" , alpha_prime_1_r_prime)

alpha_double_prime_1_r_double_prime = pow(int(g1), int(shij))
print ("α''/1r''=" , alpha_double_prime_1_r_double_prime)

beta_1_t = pow(int(h1), int(sf_prime_ij))
print ("β/1t=" , beta_1_t)

beta_prime_1_t_prime = pow(int(h2), int(sg_prime_ij))
print ("β'/1t'=" , beta_prime_1_t_prime)

beta_double_prime_1_t_double_prime = pow(int(h1), int(sh_prime_ij))
print ("β''/1t''=" , beta_double_prime_1_t_double_prime)
##############equation 3 ##############
alpha_1r = 49
beta_1t = 9
alpha_prime_1_r_prime = 25
beta_prime_1t_prime = 343

produit= alpha_1r * beta_1t * alpha_prime_1_r_prime * beta_prime_1t_prime
equation=pow(int(g1), int(sfij)) * pow(int(h1), int(sf_prime_ij))* pow(int(g2), int(sgij))* pow(int(g2), int(sg_prime_ij))
if   produit == equation :
     print("L'équation est vérifiée.")
else:
    print("L'équation n'est pas vérifiée.")
    
    
    
alpha_double_prime_1_r_double_prime=343
beta_double_prime_1_t_double_prime=243

produit1= alpha_double_prime_1_r_double_prime* beta_double_prime_1_t_double_prime
equation1=pow(int(g1), int(shij))*pow(int(h1), int(sh_prime_ij))
if   produit1 == equation1 :
     print("L'équation est vérifiée.")
else:
    print("L'équation n'est pas vérifiée.")
    
########################4-f################"
 # Définir les constantes λ2 et g1
lambda_2 = 580
g1 = 2.0

# Définir les coefficients aj pour chaque participant Pj
aj_list = [1.0, 2.0, 3.0, 4.0, 5.0]

# Définir la valeur initiale de sfij
sfij = 1.0

# Calculer sfij pour chaque participant Pj
for aj in aj_list:
    sfij = lambda_2 / (g1 * sfij) * aj
    print("sfij pour Pj =", aj, ":", sfij)
    
    # Définir les constantes λ'2 et g2
lambda_prime_2 = 1911
g2 = 3.0

# Définir les coefficients a'j pour chaque participant Pj
a_prime_j_list = [1.5, 2.5, 3.5, 4.5, 5.5]

# Définir la valeur initiale de sgij
sgij = 1.0

# Calculer sgij pour chaque participant Pj
for a_prime_j in a_prime_j_list:
    sgij = lambda_prime_2 / (g2 * sgij) * a_prime_j
    print("sgij pour Pj =", a_prime_j, ":", sgij)
    # Définir les constantes λ''2 et g1
lambda_double_prime_2 = 0.9
g1 = 4.0

# Définir les coefficients a''j pour chaque participant Pj
a_double_prime_j_list = [1.2, 2.2, 3.2, 4.2, 5.2]

# Définir la valeur initiale de shij
shij = 1.0

# Calculer shij pour chaque participant Pj
for a_double_prime_j in a_double_prime_j_list:
    shij = lambda_double_prime_2 / (g1 * shij) * a_double_prime_j
    print("shij pour Pj =", a_double_prime_j, ":", shij)
    
    # Définir les constantes γ2 et h1
gamma_2 = 0.6
h1 = 2.5

# Définir les coefficients bj pour chaque participant Pj
bj_list = [1.8, 2.8, 3.8, 4.8, 5.8]

# Définir la valeur initiale de sf'ij
sf_prime_ij = 1.0

# Calculer sf'ij pour chaque participant Pj
for bj in bj_list:
    sf_prime_ij = gamma_2 / (h1 * sf_prime_ij) * bj
    print("sf'ij pour Pj =", bj, ":", sf_prime_ij)
    
    # Définir les constantes γ'2 et h2
gamma_prime_2 = 0.8
h2 = 3.5

# Définir les coefficients b'j pour chaque participant Pj
b_prime_j_list = [1.3, 2.3, 3.3, 4.3, 5.3]

# Définir la valeur initiale de sg'ij
sg_prime_ij = 1.0

# Calculer sg'ij pour chaque participant Pj
for b_prime_j in b_prime_j_list:
    sg_prime_ij = gamma_prime_2 / (h2 * sg_prime_ij) * b_prime_j
    print("sg'ij pour Pj =", b_prime_j, ":", sg_prime_ij)
  
  # Définir les constantes γ''2 et h1
gamma_double_prime_2 = 0.9
h1 = 2.5

# Définir les coefficients b''j pour chaque participant Pj
b_double_prime_j_list = [1.2, 2.2, 3.2, 4.2, 5.2]

# Définir la valeur initiale de sh'ij
sh_prime_ij = 1.0

# Calculer sh'ij pour chaque participant Pj
for b_double_prime_j in b_double_prime_j_list:
    sh_prime_ij = gamma_double_prime_2 / (h1 * sh_prime_ij) * b_double_prime_j
    print("sh'ij pour Pj =", b_double_prime_j, ":", sh_prime_ij)
    
  ####################4-h############
 # Définir la valeur de g1sfij
g1sfij = 3.5
aj =  5.6

result = g1sfij * aj
print("(g1sfij)* aj pour le participant :", result)
    
    # Définir la valeur de g2sgij
g2sgij = 2.5

# Définir le coefficient a pour le participant
a = 1.3

# Calculer g2sgij * a pour le participant
result = g2sgij * a
print("(g2sgij) * a pour le participant :", result)


# Définir la constante lambda2
lambda2 = 0.5

# Définir la valeur de g1sfij
g1sfij = 3.5

# Définir les coefficients aj pour chaque participant Pk dans l'ensemble Q
aj_list = [1.2, 2.3, 3.4, 4.5, 5.6]

# Calculer la multiplication entre g1sfij et aj pour chaque participant Pk dans l'ensemble Q
multiplications = []
for aj in aj_list:
    result = g1sfij * aj
    multiplications.append(result)

# Calculer e1 en multipliant lambda2 avec la somme des multiplications
e1 = lambda2 * sum(multiplications)
print("e1 :", e1)

# Définir la valeur de g2sgij
g2sgij = 2.5

# Définir le coefficient a'j pour le participant
a_prime_j = 1.3

# Calculer g2sgij * a'j pour le participant
result = g2sgij * a_prime_j
print("(g2sgij) * a'j pour le participant :", result)


# Définir la valeur de h1sf'ij
h1sf_prime_ij = 1.5

# Définir le coefficient bj pour le participant
bj = 2.3

# Calculer h1sf'ij * bj pour le participant
result = h1sf_prime_ij * bj
print("(h1sf'ij) * bj pour le participant :", result)

# Définir la valeur de h2sg'ij
h2sg_prime_ij = 0.5

# Définir le coefficient b'j pour le participant
b_prime_j = 3.2

# Calculer h2sg'ij * b'j pour le participant
result = h2sg_prime_ij * b_prime_j
print("(h2sg'ij) * b'j pour le participant :", result)

# Définir la valeur de h1sh'ij
h1sh_prime_ij = 2.5

# Définir le coefficient b''j pour le participant
b_double_prime_j = 1.8

# Calculer h1sh'ij * b''j pour le participant
result = h1sh_prime_ij * b_double_prime_j
print("(h1sh'ij) * b''j pour le participant :", result)

# Définir la constante lambda2'
lambda2_prime = 0.8

# Définir la valeur de g2sgij
g2sgij = 1.5

# Définir les coefficients a'j pour chaque participant Pk dans l'ensemble Q
a_prime_j_list = [1.2, 2.3, 3.4, 4.5, 5.6]

# Calculer la multiplication entre g2sgij et a'j pour chaque participant Pk dans l'ensemble Q
multiplications = []
for a_prime_j in a_prime_j_list:
    result = g2sgij * a_prime_j
    multiplications.append(result)

# Calculer e'1 en multipliant lambda2' avec la somme des multiplications
e1_prime = lambda2_prime * sum(multiplications)
print("e'1 :", e1_prime)

# Définir la constante lambda2''
lambda2_double_prime = 0.3

# Définir la valeur de g1shij
g1shij = 2.5

# Définir les coefficients a''j pour chaque participant Pk dans l'ensemble Q
a_double_prime_j_list = [1.2, 2.3, 3.4, 4.5, 5.6]

# Calculer la multiplication entre g1shij et a''j pour chaque participant Pk dans l'ensemble Q
multiplications = []
for a_double_prime_j in a_double_prime_j_list:
    result = g1shij * a_double_prime_j
    multiplications.append(result)

# Calculer e''1 en multipliant lambda2'' avec la somme des multiplications
e1_double_prime = lambda2_double_prime * sum(multiplications)
print("e''1 :", e1_double_prime)

# Définir la constante gamma2
gamma2 = 0.4

# Définir la valeur de h1sf'ij
h1sf_prime_ij = 1.5

# Définir les coefficients bj pour chaque participant Pk dans l'ensemble Q
bj_list = [1.2, 2.3, 3.4, 4.5, 5.6]

# Calculer la multiplication entre h1sf'ij et bj pour chaque participant Pk dans l'ensemble Q
multiplications = []
for bj in bj_list:
    result = h1sf_prime_ij * bj
    multiplications.append(result)

# Calculer e2 en multipliant gamma2 avec la somme des multiplications
e2 = gamma2 * sum(multiplications)
print("e2 :", e2)

# Définir la constante gamma2'
gamma2_prime = 0.6

# Définir la valeur de h2sg'ij
h2sg_prime_ij = 2.5

# Définir les coefficients b'j pour chaque participant Pk dans l'ensemble Q
b_prime_j_list = [1.2, 2.3, 3.4, 4.5, 5.6]

# Calculer la multiplication entre h2sg'ij et b'j pour chaque participant Pk dans l'ensemble Q
multiplications = []
for b_prime_j in b_prime_j_list:
    result = h2sg_prime_ij * b_prime_j
    multiplications.append(result)

# Calculer e'2 en multipliant gamma2' avec la somme des multiplications
e2_prime = gamma2_prime * sum(multiplications)
print("e'2 :", e2_prime)

# Définir la constante gamma2''
gamma2_double_prime = 0.2

# Définir la valeur de h1sh'ij
h1sh_prime_ij = 3.5

# Définir les coefficients b''j pour chaque participant Pk dans l'ensemble Q
b_double_prime_j_list = [1.2, 2.3, 3.4, 4.5, 5.6]

# Calculer la multiplication entre h1sh'ij et b''j pour chaque participant Pk dans l'ensemble Q
multiplications = []
for b_double_prime_j in b_double_prime_j_list:
    result = h1sh_prime_ij * b_double_prime_j
    multiplications.append(result)

# Calculer e''2 en multipliant gamma2'' avec la somme des multiplications
e2_double_prime = gamma2_double_prime * sum(multiplications)
print("e''2 :", e2_double_prime)

e1 = sfij
print("e1 :", e1)

e1_prime = sgij
print("e'1 :", e1_prime)
e1_double_prime = shij
print("e''1 :", e1_double_prime)
e2 = sf_prime_ij
print("e2 :", e2)
e2_prime = sg_prime_ij
print("e'2 :", e2_prime)
e2_double_prime = sh_prime_ij
print("e''2 :", e2_double_prime)

 
if (e1 == sfij) or (e1_prime == sgij) or  (e1_double_prime == shij) == (e2 == sf_prime_ij) or (e2_prime == sg_prime_ij) or (e2_double_prime ==sh_prime_ij) :
    print("Pk conclut que Pj a menti.")
else:
    print("Pk conclut que Pi a menti.")
    
    