import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


# grouped_data = [[data[x], data[x+1]] for x in range(0, len(data), 2)]
# axis[0].plot(grouped_data[0][0], grouped_data[0][1])
# print(grouped_data[0][0], grouped_data[0][1])

# axis.plot(df[0][1:], df[1][1:])
# fig = plt.figure("first")
# axis.plot(df[2][1:], df[3][1:])

data = np.loadtxt('Data/SpectralData_Hbeta.csv', skiprows=1, delimiter=',', unpack=True)


grouped_data = [[data[x][1:], data[x+1][1:]] for x in range(0, len(data)-1, 2)]
df = pd.DataFrame(grouped_data, columns=["Frequency x-axis", "Intensity y-axis"])
# print(df)


def gaussian(freq, a, mu, sigma):
    return a*np.exp(-(freq-mu)**2/(2*sigma**2))

def line(freq, c, m, a, mu, sigma):
    return c + freq*m + gaussian(freq, a, mu, sigma)



row = 1

fig, axis = plt.subplots(1, 1)
freq, intensity = df["Frequency x-axis"][row], df["Intensity y-axis"][row]


axis.plot(freq, intensity)

#for x in range(0, 59, 2):
#    axis.plot(df["Frequency x-axis"][x], df["Intensity y-axis"][x+1])
#    fig = plt.figure(f"Number {x}")



# This function kept giving errors for my mu guess, I don't know why
def mu(x, y):
    return x[np.where(y == np.amax(y))][0]


def gradient(x, y):
    return (np.mean(y[-20:-1]) - np.mean(y[0:20])) / (x[-1] - x[0])


#mu_guess = np.mean(freq)
# logic to guess the mu
mu_guess1 = mu(freq, intensity)
grad_guess = gradient(freq, intensity)
line_fit = curve_fit(line, freq, intensity, p0=[max(intensity), grad_guess, max(intensity)*10, mu_guess1, 0.4e14], maxfev = 120000)

line = line(freq, *line_fit[0])

print(f'line = {line_fit[0][1]}x + {line_fit[0][0]}')
print(f'gaussian = {line_fit[0][2]}exp((x-{line_fit[0][3]})^2 / (2*{line_fit[0][4]}^2))')

plt.plot(freq, line)

axis.set_xlabel('Frequency/THz')
axis.set_ylabel('Intensity/A.U.')

axis.grid()




plt.show()

