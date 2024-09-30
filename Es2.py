import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

#Question 1 Find the minimum safety factor Fmin,C with varying r.
    #Definition of variables

# Constants
H = 30
L = 60
g = 9810  # specific weight of water
gsec = 19100  # gamma sec - weight of dry soil
n = 0.25
gsat = gsec + n * g
phi1 = np.pi / 6
c1 = 11400
tgtheta = 0.3
Nstr = 32  # assumption
delta = 1  # thickness

xc = 20  # assigned
zc = 50  # assigned

# Calculation of the parameters of the simplified Bishop method equation
f0 = 1 + H**2 / L**2
f1 = xc + zc * H / L
f2 = zc - xc * H / L

# Radius
r_est_inf = f2 / np.sqrt(f0)
r_est_sup = np.max([np.sqrt((xc + L / 2)**2 + zc**2), np.sqrt((xc - 3 * L / 2)**2 + (zc - H)**2)])

def x_min(r):  # abscissa initial point x_min
    if xc**2 + zc**2 > r**2:
        return (f1 - np.sqrt(r**2 * f0 - f2**2)) / f0
    elif xc**2 + zc**2 == r**2:
        return 0
    else:
        return xc - np.sqrt(r**2 - zc**2)

def x_max(r):  # abscissa end point x_max
    if (xc - L)**2 + (zc - H)**2 > r**2:
        return (f1 + np.sqrt(r**2 * f0 - f2**2)) / f0
    elif (xc - L)**2 + (zc - H)**2 == r**2:
        return L
    else:
        return xc + np.sqrt(r**2 - (zc - H)**2)

def b(r):
    return (x_max(r) - x_min(r)) / Nstr  # stripe width b

def sf(r):
    def f(F):
        x_Ii = np.zeros(Nstr)
        x_Fi = np.zeros(Nstr)
        zp_Ii = np.zeros(Nstr)
        zp_Fi = np.zeros(Nstr)
        zs_Ii = np.zeros(Nstr)
        zs_Fi = np.zeros(Nstr)
        zf_Ii = np.zeros(Nstr)
        zf_Fi = np.zeros(Nstr)
        sin_alpha_i = np.zeros(Nstr)
        cos_alpha_i = np.zeros(Nstr)
        tan_alpha_i = np.zeros(Nstr)
        zf_Mi = np.zeros(Nstr)
        V_Tsec_i = np.zeros(Nstr)
        V_Tsat_i = np.zeros(Nstr)
        U_bi = np.zeros(Nstr)
        P = 0
        Wi = np.zeros(Nstr)
        Ai = np.zeros(Nstr)
        Bi = np.zeros(Nstr)
        sum = 0

        for i in range(Nstr):
            x_Ii[i] = x_min(r) + i * b(r)  # initial abscissae of the stripe x_Ii
            x_Fi[i] = x_min(r) + (i + 1) * b(r)  # final abscissar of the stripes x_Fi
            if x_Ii[i] <= 0:
                zp_Ii[i] = 0  # initial slope heights of the stripes zp_Ii
            elif 0 < x_Ii[i] < L:
                zp_Ii[i] = x_Ii[i] * H / L
            else:
                zp_Ii[i] = H
            if x_Fi[i] <= 0:
                zp_Fi[i] = 0  # final slope heights of the stripes zp_Fi
            elif 0 < x_Fi[i] < L:
                zp_Fi[i] = x_Fi[i] * H / L
            else:
                zp_Fi[i] = H

            zs_Ii[i] = zc - np.sqrt(r**2 - (x_Ii[i] - xc)**2)  # upper quotas initial scrolling of the stripes zs_Ii
            zs_Fi[i] = zc - np.sqrt(r**2 - (x_Fi[i] - xc)**2)  # upper quotas final scrolling of the stripes zs_Fi
            zf_Ii[i] = x_Ii[i] * tgtheta  # upper quotas of the initial edge of the stripes zf_Ii
            zf_Fi[i] = x_Fi[i] * tgtheta  # upper quotas of the final layer of the stripes zf_Fi
            zf_Mi[i] = (zf_Fi[i] + zf_Ii[i]) / 2
            sin_alpha_i[i] = (zs_Fi[i] - zs_Ii[i]) / np.sqrt(b(r)**2 + (zs_Fi[i] - zs_Ii[i])**2)  # sines of the corners at the bases of the stripes sin_alpha_i
            cos_alpha_i[i] = b(r) / np.sqrt(b(r)**2 + (zs_Fi[i] - zs_Ii[i])**2)  # cosines of the angles at the bases of the stripes cos_alpha_i
            tan_alpha_i[i] = (zs_Fi[i] - zs_Ii[i]) / b(r)  # tangents of the corners to the bases of the stripes tan_alpha_i
            if zf_Mi[i] > (zs_Ii[i] + zs_Fi[i]) / 2:  # volumes of dry soil in the stripes V_Tsec_i
                V_Tsec_i[i] = delta * b(r) * (((zp_Ii[i] + zp_Fi[i]) / 2) - zf_Mi[i])
            else:
                V_Tsec_i[i] = delta * b(r) * (((zp_Ii[i] + zp_Fi[i]) / 2) - ((zs_Ii[i] + zs_Fi[i]) / 2))
            if zf_Mi[i] > (zs_Ii[i] + zs_Fi[i]) / 2:  # volumes of saturated soil in the stripes V_Tsat_i
                V_Tsat_i[i] = delta * b(r) * (zf_Mi[i] - ((zs_Ii[i] + zs_Fi[i]) / 2))
            else:
                V_Tsat_i[i] = 0
            if zf_Mi[i] > (zs_Ii[i] + zs_Fi[i]) / 2:  # modules of the pore pressure resultants agents along the bases of the stripes U_bi
                U_bi[i] = delta * (b(r) / cos_alpha_i[i]) * g * (zf_Mi[i] - ((zs_Ii[i] + zs_Fi[i]) / 2))
            else:
                U_bi[i] = 0

            # Calculation of the safety factor Fmin,C
            Wi[i] = gsec * V_Tsec_i[i] + gsat * V_Tsat_i[i]  # Stripe weights Wi
            P += Wi[i] * sin_alpha_i[i]
            Ai[i] = (c1 * b(r) / cos_alpha_i[i]) + ((Wi[i] / cos_alpha_i[i]) - U_bi[i]) * np.tan(phi1)
            Bi[i] = tan_alpha_i[i] * np.tan(phi1)
            sum += Ai[i] / (Bi[i] + F)
        return P - sum

    F = optimize.brentq(f, 0.1, 10)
    return F

r_min = optimize.minimize_scalar(sf, bounds=(r_est_inf + 1, r_est_sup), method="bounded")

print("Question n.1")
print("The minimum value for the safety factor Fmin,c according to the Simplified Bishop Method, given the center C(xc =", xc, "m, zc =", zc, "m) is F =", float("{:.3f}".format(sf(r_min.x))))
print("The radius rmin,c of the safety factor, given the center of the sliding surface, is: ", float("{:.3f}".format(r_min.x)), "m")

#Question 2 Coordinates of the relative minimum and minimum safety factor
def f12(rc):
    return rc[1] + rc[2] * H / L

def f22(rc):
    return rc[2] - rc[1] * H / L

def x_min2(rc):
    if rc[1]**2 + rc[2]**2 > rc[0]**2:
        return (f12(rc) - np.sqrt(rc[0]**2 * f0 - f22(rc)**2)) / f0
    elif rc[1]**2 + rc[2]**2 < rc[0]**2:
        return rc[1] - np.sqrt(rc[0]**2 - rc[2]**2)
    else:
        return 0

def x_max2(rc):
    if (rc[1] - L)**2 + (rc[2] - H)**2 > rc[0]**2:
        return (f12(rc) + np.sqrt(rc[0]**2 * f0 - f22(rc)**2)) / f0
    elif (rc[1] - L)**2 + (rc[2] - H)**2 < rc[0]**2:
        return rc[1] + np.sqrt(rc[0]**2 - (rc[2] - H)**2)
    else:
        return L

def b2(rc):
    return (x_max2(rc) - x_min2(rc)) / Nstr  # stripe width b

def sf2(rc):
    def f2(F):
        x_Ii = np.zeros(Nstr)
        x_Fi = np.zeros(Nstr)
        zp_Ii = np.zeros(Nstr)
        zp_Fi = np.zeros(Nstr)
        zs_Ii = np.zeros(Nstr)
        zs_Fi = np.zeros(Nstr)
        zf_Ii = np.zeros(Nstr)
        zf_Fi = np.zeros(Nstr)
        sin_alpha_i = np.zeros(Nstr)
        cos_alpha_i = np.zeros(Nstr)
        tan_alpha_i = np.zeros(Nstr)
        zf_Mi = np.zeros(Nstr)
        V_Tsec_i = np.zeros(Nstr)
        V_Tsat_i = np.zeros(Nstr)
        U_bi = np.zeros(Nstr)
        P = 0
        Wi = np.zeros(Nstr)
        Ai = np.zeros(Nstr)
        Bi = np.zeros(Nstr)
        sum_A = 0

        for i in range(Nstr):
            x_Ii[i] = x_min2(rc) + i * b2(rc)  # initial abscissae of the stripe x_Ii
            x_Fi[i] = x_min2(rc) + (i + 1) * b2(rc)  # final abscissae of the stripes x_Fi
            if x_Ii[i] <= 0:
                zp_Ii[i] = 0  # initial slope heights of the stripes zp_Ii
            elif 0 < x_Ii[i] < L:
                zp_Ii[i] = x_Ii[i] * H / L
            else:
                zp_Ii[i] = H
            if x_Fi[i] <= 0:
                zp_Fi[i] = 0  # final slope heights of the stripes zp_Fi
            elif 0 < x_Fi[i] < L:
                zp_Fi[i] = x_Fi[i] * H / L
            else:
                zp_Fi[i] = H

            zs_Ii[i] = rc[2] - np.sqrt(rc[0]**2 - (x_Ii[i] - rc[1])**2)  # upper quotas initial scrolling of the stripes zs_Ii
            zs_Fi[i] = rc[2] - np.sqrt(rc[0]**2 - (x_Fi[i] - rc[1])**2)  # upper quotas final scrolling of the stripes zs_Fi
            zf_Ii[i] = x_Ii[i] * tgtheta  # upper quotas of the initial edge of the stripes zf_Ii
            zf_Fi[i] = x_Fi[i] * tgtheta  # upper quotas of the final layer of the stripes zf_Fi
            zf_Mi[i] = (zf_Fi[i] + zf_Ii[i]) / 2
            sin_alpha_i[i] = (zs_Fi[i] - zs_Ii[i]) / np.sqrt(b2(rc)**2 + (zs_Fi[i] - zs_Ii[i])**2)  # sines of the corners at the bases of the stripes sin_alpha_i
            cos_alpha_i[i] = b2(rc) / np.sqrt(b2(rc)**2 + (zs_Fi[i] - zs_Ii[i])**2)  # cosines of the angles at the bases of the stripes cos_alpha_i
            tan_alpha_i[i] = (zs_Fi[i] - zs_Ii[i]) / b2(rc)  # tangents of the corners to the bases of the stripes tan_alpha_i
            if zf_Mi[i] > (zs_Ii[i] + zs_Fi[i]) / 2:  # volumes of dry soil in the stripes V_Tsec_i
                V_Tsec_i[i] = delta * b2(rc) * (((zp_Ii[i] + zp_Fi[i]) / 2) - zf_Mi[i])
            else:
                V_Tsec_i[i] = delta * b2(rc) * (((zp_Ii[i] + zp_Fi[i]) / 2) - ((zs_Ii[i] + zs_Fi[i]) / 2))
            if zf_Mi[i] > (zs_Ii[i] + zs_Fi[i]) / 2:  # volumes of saturated soil in the stripes V_Tsat_i
                V_Tsat_i[i] = delta * b2(rc) * (zf_Mi[i] - ((zs_Ii[i] + zs_Fi[i]) / 2))
            else:
                V_Tsat_i[i] = 0
            if zf_Mi[i] > (zs_Ii[i] + zs_Fi[i]) / 2:  # modules of the pore pressure resultants agents along the bases of the stripes U_bi
                U_bi[i] = delta * (b2(rc) / cos_alpha_i[i]) * g * (zf_Mi[i] - ((zs_Ii[i] + zs_Fi[i]) / 2))
            else:
                U_bi[i] = 0

            # Calculation of the safety factor Fmin,C
            Wi[i] = gsec * V_Tsec_i[i] + gsat * V_Tsat_i[i]  # Stripe weights Wi
            P += Wi[i] * sin_alpha_i[i]
            Ai[i] = (c1 * b2(rc) / cos_alpha_i[i]) + ((Wi[i] / cos_alpha_i[i]) - U_bi[i]) * np.tan(phi1)
            Bi[i] = tan_alpha_i[i] * np.tan(phi1)
            sum_A += Ai[i] / (Bi[i] + F)
        return P - sum_A

    F_rc2 = optimize.brentq(f2, 1, 10)
    return F_rc2

bounds = ((r_est_inf + 1, r_est_sup), (0, 25), (20, 80))
result = optimize.minimize(sf2, np.array([r_min.x, xc, zc]))

optimal_rc = result.x
optimal_F = sf2(optimal_rc)

print("Question n.2")
print(f"The overall minimum safety factor, F,is = {optimal_F:.3f}")
print(f"for the optimized radius r = {optimal_rc[0]:.3f} m, xc = {optimal_rc[1]:.3f} m, zc = {optimal_rc[2]:.3f} m.")

#Question #3
def f13(rc3):
    return rc3[1] + rc3[2] * H / L

def f23(rc3):
    return rc3[2] - rc3[1] * H / L

def x_min3(rc3):
    if rc3[1]**2 + rc3[2]**2 > rc3[0]**2:
        return (f13(rc3) - np.sqrt(rc3[0]**2 * f0 - f23(rc3)**2)) / f0
    elif rc3[1]**2 + rc3[2]**2 < rc3[0]**2:
        return rc3[1] - np.sqrt(rc3[0]**2 - rc3[2]**2)
    else:
        return 0

def x_max3(rc3):
    if (rc3[1] - L)**2 + (rc3[2] - H)**2 > rc3[0]**2:
        return (f13(rc3) + np.sqrt(rc3[0]**2 * f0 - f23(rc3)**2)) / f0
    elif (rc3[1] - L)**2 + (rc3[2] - H)**2 == rc3[0]**2:
        return L
    else:
        return rc3[1] + np.sqrt(rc3[0]**2 - (rc3[2] - H)**2)

def b3(rc3):
    return (x_max3(rc3) - x_min3(rc3)) / Nstr

def sf3(rc3):
    def f3(F):
        x_Ii = np.zeros(Nstr)
        x_Mi = np.zeros(Nstr)
        x_Fi = np.zeros(Nstr)
        zp_Ii = np.zeros(Nstr)
        zp_Fi = np.zeros(Nstr)
        zs_Ii = np.zeros(Nstr)
        zs_Fi = np.zeros(Nstr)
        zf_Ii = np.zeros(Nstr)
        zf_Fi = np.zeros(Nstr)
        sin_alpha_i = np.zeros(Nstr)
        cos_alpha_i = np.zeros(Nstr)
        tan_alpha_i = np.zeros(Nstr)
        zf_Fi = np.zeros(Nstr)
        V_Tsec_i = np.zeros(Nstr)
        V_Tsat_i = np.zeros(Nstr)
        U_bi = np.zeros(Nstr)
        Wi = np.zeros(Nstr)
        P = 0
        Ai = np.zeros(Nstr)
        Bi = np.zeros(Nstr)
        total_sum = 0

        for i in range(Nstr):
            x_Ii[i] = x_min3(rc3) + (i - 1 + 1) * b3(rc3)
            x_Mi[i] = x_min3(rc3) + (i - 0.5 + 1) * b3(rc3)
            x_Fi[i] = x_min3(rc3) + (i + 1) * b3(rc3)
            if x_Ii[i] <= 0:
                zp_Ii[i] = 0
            elif 0 < x_Ii[i] < L:
                zp_Ii[i] = x_Ii[i] * H / L
            else:
                zp_Ii[i] = H

            if x_Fi[i] <= 0:
                zp_Fi[i] = 0
            elif 0 < x_Fi[i] < L:
                zp_Fi[i] = x_Fi[i] * H / L
            else:
                zp_Fi[i] = H

            zs_Ii[i] = rc3[2] - np.sqrt(rc3[0]**2 - (x_Ii[i] - rc3[1])**2)
            zs_Fi[i] = rc3[2] - np.sqrt(rc3[0]**2 - (x_Fi[i] - rc3[1])**2)
            zf_Ii[i] = x_Ii[i] * tgO[m]
            zf_Fi[i] = x_Fi[i] * tgO[m]
            sin_alpha_i[i] = (zs_Fi[i] - zs_Ii[i]) / (np.sqrt(b3(rc3)**2 + (zs_Fi[i] - zs_Ii[i])**2))
            cos_alpha_i[i] = b3(rc3) / (np.sqrt(b3(rc3)**2 + (zs_Fi[i] - zs_Ii[i])**2))
            tan_alpha_i[i] = (zs_Fi[i] - zs_Ii[i]) / b3(rc3)
            zf_Fi[i] = (zf_Fi[i] + zf_Ii[i]) / 2

            if zf_Fi[i] > ((zs_Ii[i] + zs_Fi[i]) / 2):
                V_Tsec_i[i] = delta * b3(rc3) * ((zp_Ii[i] + zp_Fi[i]) / 2 - zf_Fi[i])
                V_Tsat_i[i] = delta * b3(rc3) * (zf_Fi[i] - (zs_Ii[i] + zs_Fi[i]) / 2)
                U_bi[i] = (delta * b3(rc3) / cos_alpha_i[i]) * g * (zf_Fi[i] - (zs_Ii[i] + zs_Fi[i]) / 2)
            else:
                V_Tsec_i[i] = delta * b3(rc3) * ((zp_Ii[i] + zp_Fi[i]) / 2 - (zs_Ii[i] + zs_Fi[i]) / 2)
                V_Tsat_i[i] = 0
                U_bi[i] = 0

            Wi[i] = gsec * V_Tsec_i[i] + gsat * V_Tsat_i[i]
            P += Wi[i] * sin_alpha_i[i]
            Ai[i] = (c1 * b3(rc3) / cos_alpha_i[i]) + (Wi[i] / cos_alpha_i[i] - U_bi[i]) * np.tan(phi1)
            Bi[i] = tan_alpha_i[i] * np.tan(phi1)
            total_sum += Ai[i] / (Bi[i] + F)

        return P - total_sum

    F_tgO = optimize.brentq(f3, 1, 10, xtol=0.0001)
    return F_tgO

# Conditions
print("Question #3")
M = 11
tgOinitial = 0.2
tgOfinal = 0.4
tgO = np.zeros(M)
plot = np.zeros(M)

for m in range(M):
    tgO[m] = tgOinitial + m * (tgOfinal - tgOinitial) / (M - 1)
    r3_xc3_zc_3 = optimize.minimize(sf3, np.array([50, 10, 40]), options={'tol': 0.0001})
    min_F = sf3(r3_xc3_zc_3.x)
    print(f"The absolute minimum safety factor for tgθ = {tgO[m]:.3f} is: {min_F:.3f}")
    print(f"obtained for r = {r3_xc3_zc_3.x[0]:.3f} m, xc = {r3_xc3_zc_3.x[1]:.3f} m, zc = {r3_xc3_zc_3.x[2]:.3f} m")
    plot[m] = min_F

plt.plot(tgO, plot, marker='o', color="blue")
plt.xlabel("Tangent of Theta, tgθ", fontsize=13)
plt.ylabel("Safety Coefficient, Fmin", fontsize=13)
plt.xlim(0.2, 0.42)
plt.ylim(1, 1.6)
plt.grid()
plt.show()