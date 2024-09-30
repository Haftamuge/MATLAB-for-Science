import numpy as np
from scipy import integrate
from scipy import optimize
from matplotlib import pyplot as plt

# Defining the given constants
g = 9.81
a = 160
b = 3.12
zfs = 354.5
qeb = 30
qec = 475
Tc = 43200
zcd = 354
Ad = 1.7
uf = 0.5
zcs = 394.6
uso = 0.48
eta = 0.12
zmi = 397.5
zmi1 = 397

#Defining the reservoir routing equation
def fqe(t):
    if t >= Tc:
        qe = qeb + (qec - qeb) * np.exp(-1.386 * (t / Tc - 1))
    else:

        qe = qeb + (qec - qeb) * (np.sin(np.pi * t / (2 * Tc)))**2
    return qe

def fquf(z):
    return np.sqrt(2 * g) * (uf * Ad * (z - zcd)**(1/2))

def fus(z):
    return uso * ((z - zcs) / (zmi - zcs)) ** eta

def fqus(z):
    if z > zcs:
        qus = np.sqrt(2 * g) * (uso * L0 * (z - zcs)**(3/2))
    else:
        qus = 0
    return qus

def fqu(z):
    return fquf(z) + fqus(z)

def f(z, t):
    return (fqe(t) - fqu(z)) / (a * b * (z - zfs) ** (b - 1))

def fqusL(z, L):
    if z > zcs:
        qus = np.sqrt(2*g)*(fus(z)*L*(z - zcs)**(3/2))
    else:
        qus = 0
    return qus

def fqusL_max(z):
    if z > zcs:
        qus = np.sqrt(2*g)*(fus(z)*L_zmax*(z - zcs)**(3/2))
    else:
        qus = 0
    return qus

def fquL(z, L):
    return fqusL(z, L) + fquf(z)

def fquL_max(z):
    return fqusL_max(z) + fquf(z)

def fzL(z, t):
    return (fqe(t) - fquL_max(z)) / (a * b * (z - zfs)**(b-1))

def f3(L):
    def f2(z, t):

        return (fqe(t) - fquL(z, L)) / (a * b * (z - zfs) ** (b - 1))

    t = np.linspace(0, 4 * Tc, 1800)
    zL_t = integrate.odeint(f2, zcs, t)
    zL_max = np.max(zL_t[:, 0])
    return zL_max - zmi

def fz0(z0):
    def f4(z, t):
        return (fqe(t) - fquL_max(z)) / (a * b * (z - zfs)**(b-1))

    t = np.linspace(0, 4*Tc, 1800)
    z_z0 = integrate.odeint(f4, z0, t)
    z_z0max = np.max(z_z0)
    return z_z0max - zmi1

# Question 1
    # Computing the initial conditions
tqe = np.linspace(0, 4 * Tc, 1800)
qe = np.zeros(1800)
for i in range(1800):
    qe[i] = fqe(tqe[i])
qe_max = np.max(qe)

Wp = (0.5 + 1/1.386) * (qec - qeb) * Tc
Wl = a * ((zmi - zfs)**b - (zcs - zfs)**b)
qu_max = (qe_max - qeb) * (1 - Wl / Wp)
L0 = qu_max / (uso * (zmi - zcs) ** (3/2) * np.sqrt(2 * g))

    # Verification of the maximum reservoir level
t = np.linspace(0, 4 * Tc, 1800)
z_t = integrate.odeint(f, zcs, t)
z_max = np.max(z_t)

print("Question #1")
print(f"The assumed value of L0 used to calculate the first zmax is: {L0:.2f} m")
print(f"The z max found for a value of L=L0 is: {z_max:.2f} m")
print(f"The compliance with the maximum reservoir quota is not fulfilled since zmax = {z_max:.2f} m.a.s.l is greater than the quota zmi= {zmi} m.a.s.l. Therefore, L needs to be changed.")

#Question 2. Optimal sizing of length L
L_zmax = optimize.brentq(f3, L0/10, L0*10)
z_Lt = integrate.odeint(fzL, zcs, t)
z_maxL = np.max(z_Lt)
t_max = t[np.argmax(z_Lt)]

print("Question #2")
print(f"Optimal sizing: L = {L_zmax:.2f} m")
print(f"Therefore, the width of the spillway must be increased from the initial attempt value by: {L_zmax - L0:.2f} m")

    #Plotting
plt.plot(t / 3600, z_Lt)
plt.xlabel("Time, hours")
plt.ylabel("Elevation z(t), m")
plt.xlim(0, 48)
plt.grid()
plt.show()

plt.plot(t / 3600, qe, label="Inflow, m^3/s")
qu = np.zeros(1800)
qus = np.zeros(1800)
quf = np.zeros(1800)
for i in range(1800):
    qu[i] = fquL_max(z_Lt[i])
    qus[i] = fqusL_max(z_Lt[i])
    quf[i] = fquf(z_Lt[i])

plt.plot(t / 3600, qu, label="Outflow m^3/s")
plt.xlabel("Time, hours")
plt.ylabel("Flow q(t), m^3/s")
plt.xlim(0, 48)
plt.legend()
plt.grid()
plt.show()

#Question 3. Reduction
res = optimize.minimize_scalar(lambda z0: np.abs(fz0(z0) - 0), bounds=(360, 395), method='bounded')
z0 = res.x
z_Lt3 = integrate.odeint(fzL, z0, t)
z_maxL3 = np.max(z_Lt3)
t_max3 = t[np.argmax(z_Lt3)]

print("Question #3")
print("The level of maximum regulation to ensure the condition z'mi = 397.0 m (fixing L calculated in the previous point) is: z0 =", float("{:.2f}".format(z0)), "[m]")
print("Therefore, the initial level to have a reduction in maximum regulation must be equal to:", float("{:.2f}".format(z0)), "m.a.s.l")
print("The reduction in the maximum regulation level is therefore:", float("{:.2f}".format(zcs - z0)))

# Plotting
plt.plot(t/3600, z_Lt3)
plt.xlabel("Time, hours")
plt.ylabel("Level z(t), m")
plt.xlim(0, 48)
plt.grid()
plt.show()

