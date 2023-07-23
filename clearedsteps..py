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

p = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109,
     113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199}

g1 = random.randint(1, len(p)-1)
g2 = random.randint(1, len(p)-1)
print(f"g1 = {g1}")
print(f"g2 = {g2}")

h1 = pow(g1, gamma1)
print("La valeur de h1 est:", h1)

h2 = pow(g2, gamma2)
print("La valeur de h2 est:", h2)

# Compute private key sk
x1 = random.randint(1, 100)
x2 = random.randint(1, 100)
y1 = random.randint(1, 100)
y2 = random.randint(1, 100)
z = random.randint(1, 100)

sk = (x1, x2, y1, y2, z)

print("sk =", sk)

# Calculer E1i, E2i, E3i, E4i et E5i
E1i = pow(g1, x1i) * pow(h1, beta1i)
E2i = pow(g1, y1i) * pow(h1, beta2i)
E3i = pow(g2, x2i) * pow(h2, beta3i)
E4i = pow(g2, y2i) * pow(h2, beta4i)
E5i = pow(g1, zi)
print("La valeur de E1i est :", E1i)
print("La valeur de E2i est :", E2i)
print("La valeur de E3i est :", E3i)
print("La valeur de E4i est :", E4i)
print("La valeur de E5i est :", E5i)

# Diffuser les valeurs E1i, E2i, E3i, E4i et E5i
broadcast_data = (E1i, E2i, E3i, E4i, E5i)

t = 5


import numpy as np

# Define the value of t
t = 5

# Generate random coefficients for the polynomials
a_coef = np.random.rand(t+1)*10
b_coef = np.random.rand(t)*10
a_prime_coef = np.random.rand(t+1)*10
b_prime_coef = np.random.rand(t)*10
a_prime_prime_coef = np.random.rand(t+1)*10
b_prime_prime_coef = np.random.rand(t)*10

# Create polynomial objects using the generated coefficients
f = np.poly1d(a_coef)
f_prime = np.poly1d(b_coef)
g = np.poly1d(a_prime_coef)
g_prime = np.poly1d(b_prime_coef)
h = np.poly1d(a_prime_prime_coef)
h_prime = np.poly1d(b_prime_prime_coef)



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


t = 5
y2i = np.random.randint(-10, 10)

# Génère aléatoirement les coefficients des termes du polynôme
g_coef = np.random.rand(t+1)*10
g_coef[0] = y2i - sum(g_coef[1:]*(-2)**np.arange(1, t+1))

# Crée un objet de polynôme à partir des coefficients générés
g = np.poly1d(g_coef)

print("Le polynôme g(x) est :", g)
print("g(-2) =", g(-2))



t = 5
beta3i = np.random.randint(-10, 10)

# Génère aléatoirement les coefficients des termes du polynôme
g_prime = np.random.rand(t+1)*10
g_prime[0] = beta3i - sum(g_prime[1:]*(-2)**np.arange(1, t+1))

# Crée un objet de polynôme à partir des coefficients générés
g = np.poly1d(g_prime)

print("Le polynôme g(x) est :", g)
print("g'(-1) =", g(-1))

# Print the polynomials
print("Le polynôme f(x) est :", f)
print("Le polynôme f'(x) est :", f_prime)
print("Le polynôme g(x) est :", g)
print("Le polynôme g'(x) est :", g_prime)
print("Le polynôme h(x) est :", h)
print("Le polynôme h'(x) est :", h_prime)

# Define the value of zi, beta0i, and participants
zi = np.random.randint(-10, 10)
beta0i = np.random.randint(-10, 10)
participants = ["P1", "P2", "P3"]

# Generate random coefficients for the polynomials h and h_prime
h_coef = np.random.rand(t+1)*10
h_coef[0] = zi
h_prime_coef = np.random.rand(t+1)*10
h_prime_coef[0] = beta0i

# Create polynomial objects using the generated coefficients
h = np.poly1d(h_coef)
h_prime = np.poly1d(h_prime_coef)

# Print the polynomials
print("Le polynôme h(x) est :", h)
print("Le polynôme h'(x) est :", h_prime)

# Calculate the values of CMik, cmik, and cmi0 for each k
for k in range(t+1):
    CMik = g_coef[k] * a_coef[k] * h_coef[k] * b_coef[k] * g_coef[k] * a_prime_coef[k] * h_coef[k] * b_prime_coef[k]
    print("La valeur de CMik est :", CMik)

    cmik = g_coef[k] * a_prime_prime_coef[k] * h_coef[k] * b_prime_prime_coef[k]
    print("La valeur de cmik est :", cmik)

    cmi0 = g_coef[k] * zi * h_coef[k] * beta0i
    print("La valeur de cmi0 est :", cmi0)

# Disqualify a participant from the list of qualified participants
Qtemp = ['P1', 'P2', 'P3']  # List of participants initially qualified

for participant in Qtemp:
    if participant in participants:
        Qtemp.remove(participant)
        print("Participant", participant, "has been disqualified. New list of qualified participants:", Qtemp)
    else:
        print("Participant", participant, "is not present in the list of qualified participants.")

# Compute the shares for each participant and index
shares = {}

def compute_shares(j):
    return j**2

def compute_shares_prime(j):
    return j**3

def compute_shares_double_prime(j):
    return 2*j

for i in range(1, t+1):
    for participant in participants:
        sfij = compute_shares(i)
        sf_prime_ij = compute_shares_prime(i)
        sgij = compute_shares_double_prime(i)
        sg_prime_ij = compute_shares_double_prime(i)
        shij = compute_shares(i+1)
        sh_prime_ij = compute_shares_double_prime(i+1)

        shares[(participant, i)] = (sfij, sf_prime_ij, sgij, sg_prime_ij, shij, sh_prime_ij)

# Print the shares for a specific participant and index
print(shares[("P2", 3)])
