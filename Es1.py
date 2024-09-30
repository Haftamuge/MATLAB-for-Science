import numpy as np
from scipy import linalg as la
import time #to obtain the computing time of the problem

#Question 1
#Defining variables
L = 3
H = 3

    #First Load Conditions
Fky1 = np.array([-10, -15, -20, -15, -25, -25, -15, -15, -25, -15, -25, -25, -15, -15, -20, -10])
Fkx1 = np.zeros(16)
#print("Fky1 = \n", Fky1, "shape:", Fkx1.shape)
#print("Fkx1 = \n", Fkx1, "shape:", Fky1.shape)

#Method of Joints. Calculations of the parameters for the projections of vector R.
sa = H / np.sqrt(H**2 + 4*L**2)
ca = 2*L / np.sqrt(H**2 + 4*L**2)
sb =  H / np.sqrt(H**2 + L**2)
cb =  L / np.sqrt(H**2 + L**2)

#Matrix formulations

    #Matrix of coefficients
a = np.zeros([32,32])

B = a.ravel()

indices1 = [0,4, 33, 138, 165, 336, 361, 534, 561, 732, 757, 926, 957, 1023]
indices2 = [2, 134, 199, 200, 268, 331, 397, 530, 595, 596, 664, 727, 793, 859]
indices3 = [34, 166, 232, 359, 363, 428, 557, 562, 628, 755, 759, 824, 953, 1019]
indices4 = [66, 198, 264, 327, 395, 396, 525, 594, 660, 723, 791, 792,921, 987]
indices5 = [67, 270, 463, 666]
indices6 = [98, 230, 231, 296, 300, 427, 429, 626, 627, 692, 696, 823, 825, 891]
indices7 = [99, 494, 495, 890]
indices8 = [101, 132, 297, 330, 497, 528, 693, 726, 893, 924, 990]
indices9 = [259, 462, 655, 858]
indices10 = [291, 302, 687, 698]

B[indices1] = 1
B[indices2] = cb
B[indices3] = sb
B[indices4] = -cb
B[indices5] = ca
B[indices6] = -sb
B[indices7] = sa
B[indices8] = -1
B[indices9] = -ca
B[indices10] = -sa

A = B.reshape((32,32))

#print("A = \n", A, "shape:", A.shape)
#np.savetxt("matrixA.txt", A, "%0.2f" , header = "Matrix A")

    #Vector of known terms
bFky1 = -Fky1
bFkx1 = -Fkx1
#print("bFky1 = \n", bFky1, "shape:", bFkx1.shape)
#print("bFkx1 = \n", bFkx1, "shape:", bFky1.shape)

b = np.vstack((bFkx1,bFky1)).ravel('F')
print("b = \n", b, "shape:", b.shape)

#First solution with  numpy linalg
startx1 = time.time()
x1 = np.linalg.solve(A,b)
endx1 = time.time()

print("x1 = \n", x1, "shape:", x1.shape)
print("Time linalg.solve =", endx1-startx1)
np.savetxt("x1 linalg numpy.txt", x1, "%0.2f" , header = "Unknown vectors")

#Second solution with Scipy linalg
startx2 = time.time()
x2 = la.solve(A,b)
endx2 = time.time()

print("x2 = \n", x2, "shape:", x2.shape)
np.savetxt("x2 linalg numpy.txt", x2, "%0.2f" , header = "Unknown vectors")
print("Time la.solve =", endx2-startx2)




