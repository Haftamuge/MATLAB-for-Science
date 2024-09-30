import numpy as np
from scipy import sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

#Given data
s = 0.4
E = 2.5 * 10**7
ni = 0.2
gamma = 25
p0 = 5
dpx = 5
dpy = 2.5
p1 = 100
p2 = 50
p3 = 50
p4 = 100

D = (E * s**3) / (12 * (1 - ni**2))

#Discretization
dx = 0.1
dy = 0.1
Nr = 76
Nc = 101
x = np.linspace(0, 10, Nc)
y = np.linspace(0, 7.5, Nr)
X, Y = np.meshgrid(x, y)
grid = X, Y

#Question #1
    #First Load Conditions (14a)
pq1 = np.zeros((Nr, Nc))
for i in range(Nr):  # rows
    for j in range(Nc):  # columns
        if (i <= 24) and (j <= 24):
            pq1[i, j] = gamma * s + 5
        if (i <= 24) and (j > 24) and (j <= 49):
            pq1[i, j] = gamma * s + 0.2 * (j + 1 - 1)
        if (i <= 24) and (j > 49) and (j <= 100):
            pq1[i, j] = gamma * s + 10
        if (i > 24) and (i <= 49) and (j <= 24):
            pq1[i, j] = gamma * s + 0.1 * (i - 1 + 1) + 2.5
        if (i > 24) and (i <= 49) and (j > 24) and (j <= 49):
            pq1[i, j] = gamma * s + 0.1 * (i - 1 + 1) + 0.2 * (j - 1 + 1) - 2.5
        if (i > 24) and (i <= 49) and (j > 49) and (j <= 100):
            pq1[i, j] = gamma * s + 0.1 * (i - 1 + 1) + 7.5
        if (i > 49) and (i <= 75) and (j <= 24):
            pq1[i, j] = gamma * s + 7.5
        if (i > 49) and (i <= 75) and (j > 24) and (j <= 49):
            pq1[i, j] = gamma * s + 0.2 * (j - 1 + 1) + 2.5
        if (i > 49) and (i <= 75) and (j > 49) and (j <= 100):
            pq1[i, j] = gamma * s + 12.5

    #Setting up the boundary conditions
b1 = pq1 * dx * dy  / D
for i in range(Nr):
    for j in range(Nc):
        if (i == 0) or (j == 0) or (i == (Nr - 1)) or (j == (Nc - 1)):
            b1[i, j] = b1[i, j] / 2

b1[0, :] += 0
b1[Nr - 1, :] += 0
b1[:, 0] += 0
b1[:, -1] += 0

b1 = -b1.reshape(Nr * Nc) / dx ** 2

#print("Vector b1 dimensions", b1.shape) #to check dimensions for matrix operations

    #Creating Matrix A and its Mask for Discretization
A_1d = (np.eye(Nc, k=-1) + np.eye(Nc, k=1) - 4 * np.eye(Nc)) / (dx ** 2)
A = np.kron(np.eye(Nc), A_1d) + (np.eye(Nc ** 2, k=-Nc) + np.eye(Nc ** 2, k=Nc)) / (dx ** 2)

#print("Matrix A Dimensions", A.shape)

label = np.ones((Nc, Nc))
for j in range(Nc):
    for i in range(Nc):
        if i > (Nr-1):
            label[i, j] = 0

mask_matrix = label
mask_matrix = mask_matrix.reshape(Nc ** 2)

rc_delete = np.zeros(np.count_nonzero(mask_matrix == 0), int)
counter = 0
for i in range(Nc **  2):
    if mask_matrix[i] == 0:
        rc_delete[counter] = i
        counter = counter + 1

    #Integrating Matrix A
A_int = np.delete(A, rc_delete, 0)
A_int = np.delete(A_int, rc_delete, 1)

#print("Matrix A_int Dimension", A_int.shape)

    #Solving the System of Equations
A_csr = sp.csr_matrix(A_int) #Convert to CSR
u = sp.linalg.spsolve(A_csr, b1) #Solution

#print("Matrix u Dimensions", u.shape)

w = sp.linalg.spsolve(A_csr, u)
w = -w.reshape((Nr, Nc))

#print("Dimensions Matrix w", w.shape)

w_1 = np.insert(w, 0, 0, axis=1)
w_1 = np.insert(w_1, 0, 0, axis=0 )
w_1 = np.insert(w_1, Nc + 1, 0, axis=1)
w_1 = np.insert(w_1, Nr + 1, 0, axis=0 )

    #Calculation of Bending Moments (Mxy and Myx)
Mxy = np.zeros((76, 101))
for i in range(1, 75):
    for j in range(1, 100):
        Mxy[i, j] = -(D / 4) * (((w_1[i - 1, j + 1] - 2 * w_1[i - 1, j] + w_1[i - 1, j - 1]) / (dx ** 2))
                    + ni * (w_1[i + 1, j - 1] - 2 * w_1[i, j - 1] + w_1[i - 1, j - 1]) / (dy ** 2))\
                    - (D / 2) * ((w_1[i, j + 1] - 2 * w_1[i, j] + w_1[i, j - 1]) / (dx ** 2) +
                    ni * ( w_1[i + 1, j] - 2 * w_1[i, j] + w_1[i - 1, j]) / (dy ** 2)) - (D / 4) * \
                    (((w_1[i + 1, j + 1] - 2 * w_1[i + 1, j] + w_1[i + 1, j - 1]) / (dx ** 2)) +
                    ni * (w_1[i + 1, j + 1] - 2 * w_1[i, j + 1] + w_1[i - 1, j + 1]) / (dy ** 2))

x = np.linspace(0, 10, 101)
y = np.linspace(0, 7.5, 76)
X, Y = np.meshgrid(x, y)
grid = X, Y

Myx = np.zeros((76, 101))
for i in range(1, 75):
    for j in range(1, 100):
        Myx[i, j] = -(D / 4) * (((w_1[i + 1, j - 1] - 2 * w_1[i, j - 1] + w_1[i - 1, j - 1]) / (dy ** 2))
                    + ni * (w_1[i - 1, j + 1] - 2 * w_1[i - 1, j] + w_1[i - 1, j - 1]) / (dx ** 2)) \
                    - (D / 2) * ((w_1[i + 1, j] - 2 * w_1[i, j] + w_1[i - 1, j]) / (dy ** 2)
                    + ni * (w_1[i, j + 1] - 2 * w_1[i, j] + w_1[i, j - 1]) / (dx ** 2)) \
                    - (D / 4) * (((w_1[i + 1, j + 1] - 2 * w_1[i, j + 1] + w_1[i - 1, j + 1]) / (dy ** 2)) +
                    ni * (w_1[i + 1, j + 1] - 2 * w_1[i + 1, j] + w_1[i + 1, j - 1]) / (dx ** 2))

#print("Dimensions Matrix w_1", w_1.shape)

    #Calculation of the Shear Forces (Txy and Tyx)
u_rhsp = -u.reshape((Nr, Nc))
#print("Dimensions for u_rhsp", u_rhsp.shape)

u_T = np.insert(u_rhsp, 0, 0, axis=1)
u_T = np.insert(u_T, 0, 0, axis=0 )
u_T = np.insert(u_T, Nc + 1, 0, axis=1)
u_T = np.insert(u_T, Nr + 1, 0, axis=0 )

#print("Dimensions for u_T", u_T.shape)

Txz = np.zeros((76, 101))
for i in range(76):
    for j in range(101):
        if (j == 0) and (i < 75):
            Txz[i, j] = - D * (u_T[i, j + 1] - u_T[i, j]) / dx
        if (i == 0) and (j < 100):
            Txz[i, j] = - D * (u_T[i, j + 1] - u_T[i, j]) / dx
        if (j == 100) and (i > 0):
            Txz[i, j] = - D * (u_T[i, j] - u_T[i, j - 1]) / dx
        if (i == 75) and (j > 0):
            Txz[i, j] = - D * (u_T[i, j] - u_T[i, j - 1]) / dx
        if (i == 0) and (j == 100):
            Txz[i, j] = - D * (u_T[i, j] - u_T[i, j - 1]) / dx
        if (i == 75) and (j == 0):
            Txz[i, j] = - D * (u_T[i, j + 1] - u_T[i, j]) / dx
        if (i == 0) and (j == 0):
            Txz[i, j] = - D * (u_T[i, j + 1] - u_T[i, j]) / dx
        if (1 <= i <= 74) and (1 <= j < 100):
            Txz[i, j] = -D * (u_T[i, j + 1] - u_T[i, j - 1]) / (2 * dx)

Tyz = np.zeros((76, 101))
for i in range(76):
    for j in range(101):
        if (j == 0) and (i < 75):
            Tyz[i, j] = -D * (u_T[i + 1, j] - u_T[i, j]) / dy
        if (i == 0) and (j < 100):
            Tyz[i, j] = -D * (u_T[i + 1, j] - u_T[i, j]) / dy
        if (j == 100) and (i > 0):
            Tyz[i, j] = -D * (u_T[i, j] - u_T[i - 1, j]) / dy
        if (i == 75) and (j > 0):
            Tyz[i, j] = -D * (u_T[i, j] - u_T[i - 1, j]) / dy
        if (i == 0) and (j == 100):
            Tyz[i, j] = -D * (u_T[i + 1, j] - u_T[i, j]) / dy
        if (i == 75) and (j == 0):
            Tyz[i, j] = -D * (u_T[i, j] - u_T[i - 1, j]) / dy
        if (i == 0) and (j == 0):
            Txz[i, j] = -D * (u_T[i + 1, j] - u_T[i, j]) / dy
        if (1 <= i <= 74) and (1 <= j < 100):
            Tyz[i, j] = -D * (u_T[i + 1, j] - u_T[i - 1, j]) / (2 * dy)

    #Calculation of the Torsion
    Mxx = np.zeros((76, 101))
    for i in range(76):
        for j in range(101):
            if (j == 0) and (i < 75):
                Mxx[i, j] = -D * (1 - ni) * ((w_1[i + 1, j + 1] - w_1[i + 1, j] - w_1[i, j + 1] +
                                              w_1[i, j]) / dx ** 2)
            if (i == 0) and (j < 100):
                Mxx[i, j] = -D * (1 - ni) * ((w_1[i + 1, j + 1] - w_1[i + 1, j] - w_1[i, j + 1] +
                                              w_1[i, j]) / dx ** 2)
            if (j == 100) and (i > 0):
                Mxx[i, j] = -D * (1 - ni) * ((w_1[i - 1, j - 1] - w_1[i - 1, j] - w_1[i, j - 1] +
                                              w_1[i, j]) / dx ** 2)
            if (i == 75) and (j > 0):
                Mxx[i, j] = -D * (1 - ni) * ((w_1[i - 1, j - 1] - w_1[i - 1, j] - w_1[i, j - 1] +
                                              w_1[i, j]) / dx ** 2)
            if (i == 0) and (j == 0):
                Mxx[i, j] = -D * (1 - ni) * ((w_1[i + 1, j + 1] - w_1[i + 1, j] - w_1[i, j + 1] +
                                              w_1[i, j]) / dx ** 2)
            if (i == 75) and (j == 100):
                Mxx[i, j] = -D * (1 - ni) * ((w_1[i - 1, j - 1] - w_1[i - 1, j] - w_1[i, j - 1] +
                                              w_1[i, j]) / dx ** 2)
            if (1 <= i <= 74) and (1 <= j < 100):
                Mxx[i, j] = -D * (1 - ni) * (
                        w_1[i + 1, j + 1] - w_1[i + 1, j - 1] - w_1[i - 1, j + 1] + w_1[
                    i - 1, j - 1]) / (4 * dx * dy)

#Plots for Question #1 (First Load Conditions)
    #Sags due to the first load conditions
fig = plt.figure(figsize=(12, 5.5))
cmap = mpl.colormaps.get_cmap('RdBu_r')
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolor(X, Y, w*1000, cmap=cmap)
plt.title("Sags Due to the First Load Condition [mm]")
ax.set_xlabel(r"$x_{1}$", fontsize=18)
ax.set_ylabel(r"$x_{2}$", fontsize=18)
cb2 = plt.colorbar(c, ax=ax, shrink=1)
plt.show()

    #Bending Moments
fig = plt.figure(figsize=(13, 5.5))
cmap = mpl.colormaps.get_cmap('RdBu_r')
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolor(X, Y, Mxy, cmap=cmap)
plt.title("Mxy First Load Condition [kN*m]")
ax.set_xlabel(r"$x_{1}$", fontsize=18)
ax.set_ylabel(r"$x_{2}$", fontsize=18)
cb4 = plt.colorbar(c, ax=ax, shrink=1)
plt.show()

fig = plt.figure(figsize=(13, 5.5))
cmap = mpl.colormaps.get_cmap('RdBu_r')
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolor(X, Y, Myx, cmap=cmap)
plt.title("Myx First Load Conditions [kN*m]")
ax.set_xlabel(r"$x_{1}$", fontsize=18)
ax.set_ylabel(r"$x_{2}$", fontsize=18)
cb4 = plt.colorbar(c, ax=ax, shrink=1)
plt.show()

    #Plotting the Shear Forces
fig = plt.figure(figsize=(13, 5.5))
cmap = mpl.colormaps.get_cmap('RdBu_r')
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolor(X, Y, Txz, cmap=cmap)
plt.title("Txz First Load Conditions [kN]")
ax.set_xlabel(r"$x_{1}$", fontsize=18)
ax.set_ylabel(r"$x_{2}$", fontsize=18)
cb4 = plt.colorbar(c, ax=ax, shrink=1)
plt.show()

fig = plt.figure(figsize=(13, 5.5))
cmap = mpl.colormaps.get_cmap('RdBu_r')
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolor(X, Y, Tyz, cmap=cmap)
plt.title("Tyz First Load Conditions [kN]")
ax.set_xlabel(r"$x_{1}$", fontsize=18)
ax.set_ylabel(r"$x_{2}$", fontsize=18)
cb4 = plt.colorbar(c, ax=ax, shrink=1)
plt.show()

fig = plt.figure(figsize=(13, 5.5))
cmap = mpl.colormaps.get_cmap('RdBu_r')
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolor(X, Y, Mxx, cmap=cmap)
plt.title("Mxx First Load Conditions [kN*m]")
ax.set_xlabel(r"$x_{1}$", fontsize=18)
ax.set_ylabel(r"$x_{2}$", fontsize=18)
cb4 = plt.colorbar(c, ax=ax, shrink=1)
plt.show()

#Values and Coordinates for Question #1
print("Question #1. Maximum Stresses for First Load Conditions ")

Mxymax = np.max(Mxy)
posMxymax = np.argmax(Mxy)
row_max = posMxymax // 101
column_max = posMxymax % 101
print(f"The maximum value of Mxy is: {Mxymax:.2f} kN*m")
print(f"at coordinates x = {column_max * 0.1:.2f} m and y = {row_max * 0.1:.2f} m")

Mxymin = np.min(Mxy)
posMxymin = np.argmin(Mxy)
row_min = posMxymin // 101
column_min = posMxymin % 101
print(f"The minimum value of Mxy is: {Mxymin: .2f}, kN*m")
print(f"at coordinates x = {column_min * 0.1:.2f} m and y = {row_min * 0.1:.2f} m")

Myxmax = np.max(Myx)
posMyxmax = np.argmax(Myx)
row_max = posMyxmax // 101
column_max = posMyxmax % 101
print(f"The maximum value of Myx is: {Myxmax:.2f} kN*m")
print(f"at coordinates x = {column_max * 0.1:.2f} m and y = {row_max * 0.1:.2f} m")

Myxmin = np.min(Myx)
posMyxmin = np.argmin(Myx)
row_min = posMyxmin // 101
column_min = posMyxmin % 101
print(f"The minimum value of Myx is: {Myxmin: .2f}, kN*m")
print(f"at coordinates x = {column_min * 0.1:.2f} m and y = {row_min * 0.1:.2f} m")

Mxxmax = np.max(Mxx)
posMxxmax = np.argmax(Mxx)
row_max = posMxxmax // 101
column_max = posMxxmax % 101
print(f"The maximum value of Mxx is: {Mxxmax:.2f} kN*m")
print(f"at coordinates x = {column_max * 0.1:.2f} m and y = {row_max * 0.1:.2f} m")

Mxxmin = np.min(Mxx)
posMxxmin = np.argmin(Mxx)
row_min = posMxxmin // 101
column_min = posMxxmin % 101
print(f"The minimum value of Mxx is: {Mxxmin: .2f}, kN*m")
print(f"at coordinates x = {column_min * 0.1:.2f} m and y = {row_min * 0.1:.2f} m")

Txzmax = np.max(Txz)
posTxzmax = np.argmax(Txz)
row_max = posTxzmax // 101
column_max = posTxzmax % 101
print(f"The maximum value of Txz is: {Txzmax:.2f} kN*m")
print(f"at coordinates x = {column_max * 0.1:.2f} m and y = {row_max * 0.1:.2f} m")

Txzmin = np.min(Txz)
posTxzmin = np.argmin(Txz)
row_min = posTxzmin // 101
column_min = posTxzmin % 101
print(f"The minimum value of Txz is: {Txzmin: .2f}, kN*m")
print(f"at coordinates x = {column_min * 0.1:.2f} m and y = {row_min * 0.1:.2f} m")

Tyzmax = np.max(Tyz)
posTyzmax = np.argmax(Tyz)
row_max = posTyzmax // 101
column_max = posTyzmax % 101
print(f"The maximum value of Tyz is: {Tyzmax:.2f} kN*m")
print(f"at coordinates x = {column_max * 0.1:.2f} m and y = {row_max * 0.1:.2f} m")

Tyzmin = np.min(Tyz)
posTyzmin = np.argmin(Tyz)
row_min = posTyzmin // 101
column_min = posTyzmin % 101
print(f"The minimum value of Tyz is: {Tyzmin: .2f}, kN*m")
print(f"at coordinates x = {column_min * 0.1:.2f} m and y = {row_min * 0.1:.2f} m")

#Question 2. Second Load Conditions
    #Second Load Conditions (14b)
pq2 = np.zeros((Nr, Nc))
for i in range(Nr):
    for j in range(Nc):
        if i <= 24 and j <= 24:
            pq2[i, j] = gamma * s + p0
        if i <= 24 and j > 24 and j <= 49:
            pq2[i, j] = gamma * s + 0.2 * (j + 1 - 1)
        if i <= 24 and j > 49 and j <= 100:
            pq2[i, j] = gamma * s + 10
        if i > 24 and i <= 49 and j <= 24:
            pq2[i, j] = gamma * s + 0.1 * (i - 1 + 1) + 2.5
        if i > 24 and i <= 49 and j > 24 and j <= 49:
            if i >= 24 and i <= 26 and j >= 25 and j <= 26:
                pq2[i, j] = gamma * s + 0.1 * (i - 1 + 1) + 0.2 * (j - 1 + 1) - 2.5 + p1
            else:
                pq2[i, j] = gamma * s + 0.1 * (i - 1 + 1) + 0.2 * (j - 1 + 1) - 2.5
        if i > 24 and i <= 49 and j > 49 and j <= 100:
            if i >= 24 and i <= 26 and j >= 50 and j <= 51:
                pq2[i, j] = gamma * s + 0.1 * (i - 1 + 1) + 7.5 + p3
            if i >= 24 and i <= 26 and j >= 75 and j <= 76:
                pq2[i, j] = gamma * s + 0.1 * (i - 1 + 1) + 7.5 + p4
            else:
                pq2[i, j] = gamma * s + 0.1 * (i - 1 + 1) + 7.5
        if i > 49 and i <= 75 and j <= 24:
            pq2[i, j] = gamma * s + 7.5
        if i > 49 and i <= 75 and j > 24 and j <= 49:
            if i >= 49 and i <= 51 and j >= 25 and j <= 26:
                pq2[i, j] = gamma * s + 0.2 * (j - 1 + 1) + 2.5 + p2
            else:
                pq2[i, j] = gamma * s + 0.2 * (j - 1 + 1) + 2.5
        if i > 49 and i <= 75 and j > 49 and j <= 100:
            pq2[i, j] = gamma * s + 12.5

#Setting up boundary conditions
b2 = np.zeros((76, 101))
b2 = pq2 * dx * dy  / D
for i in range(76):
    for j in range(101):
        if i == 0 or j == 0 or i == (Nr - 1) or j ==(Nc - 1):
            b2[i, j] = 0.5 * pq2[i,j]  / D
        else:
            b2[i, j] = pq2[i,j] / D

b2[0, :] += 0
b2[Nr - 1, :] += 0
b2[:, 0] += 0
b2[:, -1] += 0
b2 = -b2.reshape(Nr * Nc) / dx ** 2

u2 = sp.linalg.spsolve(A_csr, b2)
w2 = sp.linalg.spsolve(A_csr, u2)
w2 = -w.reshape((76, 101))

w_2 = np.insert(w, 0, 0, axis=1)
w_2 = np.insert(w_2, 0, 0, axis=0 )
w_2 = np.insert(w_2, Nc + 1, 0, axis=1)
w_2 = np.insert(w_2, Nr + 1, 0, axis=0 )

    #Calculation of Bending Moments (Mxy2 and Myx2)
Mxy2 = np.zeros((76, 101))
for i in range(1, 75):
    for j in range(1, 100):
        Mxy2[i, j] = -(D / 4) * (((w_2[i - 1, j + 1] - 2 * w_2[i - 1, j] + w_2[i - 1, j - 1]) / (dx ** 2))
                    + ni * (w_2[i + 1, j - 1] - 2 * w_2[i, j - 1] + w_2[i - 1, j - 1]) / (dy ** 2))\
                    - (D / 2) * ((w_2[i, j + 1] - 2 * w_2[i, j] + w_2[i, j - 1]) / (dx ** 2) +
                    ni * ( w_2[i + 1, j] - 2 * w_2[i, j] + w_2[i - 1, j]) / (dy ** 2)) - (D / 4) * \
                    (((w_2[i + 1, j + 1] - 2 * w_2[i + 1, j] + w_2[i + 1, j - 1]) / (dx ** 2)) +
                    ni * (w_2[i + 1, j + 1] - 2 * w_2[i, j + 1] + w_2[i - 1, j + 1]) / (dy ** 2))

Myx2 = np.zeros((76, 101))
for i in range(1, 75):
    for j in range(1, 100):
        Myx2[i, j] = -(D / 4) * (((w_2[i + 1, j - 1] - 2 * w_2[i, j - 1] + w_2[i - 1, j - 1]) / (dy ** 2))
                    + ni * (w_2[i - 1, j + 1] - 2 * w_2[i - 1, j] + w_2[i - 1, j - 1]) / (dx ** 2)) \
                    - (D / 2) * ((w_2[i + 1, j] - 2 * w_2[i, j] + w_2[i - 1, j]) / (dy ** 2)
                    + ni * (w_2[i, j + 1] - 2 * w_2[i, j] + w_2[i, j - 1]) / (dx ** 2)) \
                    - (D / 4) * (((w_2[i + 1, j + 1] - 2 * w_2[i, j + 1] + w_2[i - 1, j + 1]) / (dy ** 2)) +
                    ni * (w_2[i + 1, j + 1] - 2 * w_2[i + 1, j] + w_2[i + 1, j - 1]) / (dx ** 2))

    #Calculation of the Torsion
    Mxx2 = np.zeros((76, 101))
    for i in range(76):
        for j in range(101):
            if (j == 0) and (i < 75):
                Mxx2[i, j] = -D * (1 - ni) * ((w_2[i + 1, j + 1] - w_2[i + 1, j] - w_2[i, j + 1] +
                                              w_2[i, j]) / dx ** 2)
            if (i == 0) and (j < 100):
                Mxx2[i, j] = -D * (1 - ni) * ((w_2[i + 1, j + 1] - w_2[i + 1, j] - w_2[i, j + 1] +
                                              w_2[i, j]) / dx ** 2)
            if (j == 100) and (i > 0):
                Mxx2[i, j] = -D * (1 - ni) * ((w_2[i - 1, j - 1] - w_2[i - 1, j] - w_2[i, j - 1] +
                                              w_2[i, j]) / dx ** 2)
            if (i == 75) and (j > 0):
                Mxx2[i, j] = -D * (1 - ni) * ((w_2[i - 1, j - 1] - w_2[i - 1, j] - w_2[i, j - 1] +
                                              w_2[i, j]) / dx ** 2)
            if (i == 0) and (j == 0):
                Mxx2[i, j] = -D * (1 - ni) * ((w_2[i + 1, j + 1] - w_2[i + 1, j] - w_2[i, j + 1] +
                                              w_2[i, j]) / dx ** 2)
            if (i == 75) and (j == 100):
                Mxx2[i, j] = -D * (1 - ni) * ((w_2[i - 1, j - 1] - w_2[i - 1, j] - w_2[i, j - 1] +
                                              w_2[i, j]) / dx ** 2)
            if (1 <= i <= 74) and (1 <= j < 100):
                Mxx2[i, j] = -D * (1 - ni) * (
                        w_1[i + 1, j + 1] - w_2[i + 1, j - 1] - w_2[i - 1, j + 1] + w_2[
                    i - 1, j - 1]) / (4 * dx * dy)

#Plots for Question #2 (Second Load Conditions)
    #Sags due to the second load conditions
fig = plt.figure(figsize=(12, 5.5))
cmap = mpl.colormaps.get_cmap('RdBu_r')
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolor(X, Y, w*1000, cmap=cmap)
plt.title("Sags Due to the Second Load Condition [mm]")
ax.set_xlabel(r"$x_{1}$", fontsize=18)
ax.set_ylabel(r"$x_{2}$", fontsize=18)
cb2 = plt.colorbar(c, ax=ax, shrink=1)
plt.show()

    #Bending Moments
fig = plt.figure(figsize=(13, 5.5))
cmap = mpl.colormaps.get_cmap('RdBu_r')
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolor(X, Y, Mxy2, cmap=cmap)
plt.title("Mxy Second Load Condition [kN*m]")
ax.set_xlabel(r"$x_{1}$", fontsize=18)
ax.set_ylabel(r"$x_{2}$", fontsize=18)
cb4 = plt.colorbar(c, ax=ax, shrink=1)
plt.show()

fig = plt.figure(figsize=(13, 5.5))
cmap = mpl.colormaps.get_cmap('RdBu_r')
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolor(X, Y, Myx2, cmap=cmap)
plt.title("Myx Second Load Conditions [kN*m]")
ax.set_xlabel(r"$x_{1}$", fontsize=18)
ax.set_ylabel(r"$x_{2}$", fontsize=18)
cb4 = plt.colorbar(c, ax=ax, shrink=1)
plt.show()

    #Plotting the Shear Forces
fig = plt.figure(figsize=(13, 5.5))
cmap = mpl.colormaps.get_cmap('RdBu_r')
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolor(X, Y, Txz, cmap=cmap)
plt.title("Txz Second Load Conditions [kN]")
ax.set_xlabel(r"$x_{1}$", fontsize=18)
ax.set_ylabel(r"$x_{2}$", fontsize=18)
cb4 = plt.colorbar(c, ax=ax, shrink=1)
plt.show()

fig = plt.figure(figsize=(13, 5.5))
cmap = mpl.colormaps.get_cmap('RdBu_r')
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolor(X, Y, Tyz, cmap=cmap)
plt.title("Tyz Second Load Conditions [kN]")
ax.set_xlabel(r"$x_{1}$", fontsize=18)
ax.set_ylabel(r"$x_{2}$", fontsize=18)
cb4 = plt.colorbar(c, ax=ax, shrink=1)
plt.show()

fig = plt.figure(figsize=(13, 5.5))
cmap = mpl.colormaps.get_cmap('RdBu_r')
ax = fig.add_subplot(1, 1, 1)
c = ax.pcolor(X, Y, Mxx2, cmap=cmap)
plt.title("Mxx Second Load Conditions [kN*m]")
ax.set_xlabel(r"$x_{1}$", fontsize=18)
ax.set_ylabel(r"$x_{2}$", fontsize=18)
cb4 = plt.colorbar(c, ax=ax, shrink=1)
plt.show()

#Values and Coordinates for Question #2
print("Question #2. Maximum Stresses for Second Load Conditions ")

Mxymax = np.max(Mxy2)
posMxymax = np.argmax(Mxy2)
row_max = posMxymax // 101
column_max = posMxymax % 101
print(f"The maximum value of Mxy is: {Mxymax:.2f} kN*m")
print(f"at coordinates x = {column_max * 0.1:.2f} m and y = {row_max * 0.1:.2f} m")

Mxymin = np.min(Mxy2)
posMxymin = np.argmin(Mxy2)
row_min = posMxymin // 101
column_min = posMxymin % 101
print(f"The minimum value of Mxy is: {Mxymin: .2f}, kN*m")
print(f"at coordinates x = {column_min * 0.1:.2f} m and y = {row_min * 0.1:.2f} m")

Myxmax = np.max(Myx2)
posMyxmax = np.argmax(Myx2)
row_max = posMyxmax // 101
column_max = posMyxmax % 101
print(f"The maximum value of Myx is: {Myxmax:.2f} kN*m")
print(f"at coordinates x = {column_max * 0.1:.2f} m and y = {row_max * 0.1:.2f} m")

Myxmin = np.min(Myx2)
posMyxmin = np.argmin(Myx2)
row_min = posMyxmin // 101
column_min = posMyxmin % 101
print(f"The minimum value of Myx is: {Myxmin: .2f}, kN*m")
print(f"at coordinates x = {column_min * 0.1:.2f} m and y = {row_min * 0.1:.2f} m")

Mxxmax = np.max(Mxx2)
posMxxmax = np.argmax(Mxx2)
row_max = posMxxmax // 101
column_max = posMxxmax % 101
print(f"The maximum value of Mxx is: {Mxxmax:.2f} kN*m")
print(f"at coordinates x = {column_max * 0.1:.2f} m and y = {row_max * 0.1:.2f} m")

Mxxmin = np.min(Mxx2)
posMxxmin = np.argmin(Mxx2)
row_min = posMxxmin // 101
column_min = posMxxmin % 101
print(f"The minimum value of Mxx is: {Mxxmin: .2f}, kN*m")
print(f"at coordinates x = {column_min * 0.1:.2f} m and y = {row_min * 0.1:.2f} m")

Txzmax = np.max(Txz)
posTxzmax = np.argmax(Txz)
row_max = posTxzmax // 101
column_max = posTxzmax % 101
print(f"The maximum value of Txz is: {Txzmax:.2f} kN*m")
print(f"at coordinates x = {column_max * 0.1:.2f} m and y = {row_max * 0.1:.2f} m")

Txzmin = np.min(Txz)
posTxzmin = np.argmin(Txz)
row_min = posTxzmin // 101
column_min = posTxzmin % 101
print(f"The minimum value of Txz is: {Txzmin: .2f}, kN*m")
print(f"at coordinates x = {column_min * 0.1:.2f} m and y = {row_min * 0.1:.2f} m")

Tyzmax = np.max(Tyz)
posTyzmax = np.argmax(Tyz)
row_max = posTyzmax // 101
column_max = posTyzmax % 101
print(f"The maximum value of Tyz is: {Tyzmax:.2f} kN*m")
print(f"at coordinates x = {column_max * 0.1:.2f} m and y = {row_max * 0.1:.2f} m")

Tyzmin = np.min(Tyz)
posTyzmin = np.argmin(Tyz)
row_min = posTyzmin // 101
column_min = posTyzmin % 101
print(f"The minimum value of Tyz is: {Tyzmin: .2f}, kN*m")
print(f"at coordinates x = {column_min * 0.1:.2f} m and y = {row_min * 0.1:.2f} m")