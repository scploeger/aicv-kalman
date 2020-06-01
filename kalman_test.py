#By: Spencer Ploeger
#June 1 2020
#Based off of examples in: https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

import numpy as np
import matplotlib.pyplot as plt


iterations = 75

#true value
x = -2

#create random noise for each "sample" with std. dev. of 0.2
noise = np.random.normal(0, 0.2, size=iterations)

A = -1 #negative so error converges
H = 1 #=1 because the state IS the measurement

Q = 0.0005
R = 0.1 #assume a low measurement variance

#empty lists
x_hat = [0] * iterations
x_hat_minus = [0] * iterations
P = [0] * iterations
P_minus = [0] * iterations
K = [0] * iterations
z = [0] * iterations


#for first iteration assume all previous measurements are zero
k = 0 #first iteration
z[0] = x + noise[k] #measurement for this step is 

#time update
x_hat_minus[k] = 0
P_minus[k] = 0 + Q

#measurement update
K[k] = P_minus[k]/(P_minus[k] + R)
x_hat[k] = x_hat_minus[k] + K[k] * (z[0] - x_hat_minus[k])
P[k] = (1 - K[k]) * P_minus[k]

#for the rest of the iterations, loop
for k in range(1, iterations):
    z[k] = x + noise[k] #measurement for this step is 

    #time update
    x_hat_minus[k] = x_hat[k-1]
    P_minus[k] = P[k-1] + Q

    #measurement update
    K[k] = P_minus[k]/(P_minus[k] + R)
    x_hat[k] = x_hat_minus[k] + K[k] * (z[k] - x_hat_minus[k])
    P[k] = (1 - K[k]) * P_minus[k]
    
plt.figure()
plt.plot(x_hat, label = 'estimate', color = 'r')
plt.plot(z, label = 'measurement', color = 'k')
plt.axhline(y=x, label = 'ground truth')
plt.legend()
plt.show()