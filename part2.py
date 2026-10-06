import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

class Data: # this class is dedicated to gathering data from main storage and processing it for the rest of the code
    def __init__(self, path):
        self.path = path
        
    def process_data(self): # gets the main bit of data, groups the x and y columns together and exports the two 2d arrays
        data = np.loadtxt(self.path, skiprows=1, delimiter=',', unpack=True)

        # grouped_data = [[data[x][1:], data[x+1][1:]] for x in range(0, len(data)-1, 2)]

        # creates a 2d array, where each item contains the x axis data and y axis data
        grouped_data = [[data[x], data[x+1]] for x in range(0, len(data)-1, 2)]
        df = pd.DataFrame(grouped_data, columns=["Frequency x-axis", "Intensity y-axis"]) # converts to dataframe
        freq, intensity = df["Frequency x-axis"], df["Intensity y-axis"]
        # print(freq, intensity)
        return freq, intensity

    def process_distances(self):
        data = pd.read_csv(self.path, delimiter='\t') # reads the text file of distances
        distances = []
        # this for loop iterates through the data and creates an array of dictionaries for each row of data
        for x in range(len(data)):
            distances.append({'number': data['Observation_number'][x], 'distance': data['Distance'][x], 'error': data['Instrument_response'][x] })
        
        return np.array(distances)
    
    def get_valid_entries(self): # filters the array of rows to only contain valid entries
        return np.array([x for x in self.process_distances() if x['error'] == '1'])

    def get_valid_ids(self): # returns the ids of valid entries
        valid_entries = self.get_valid_entries()
        valid = np.array([x['number'] for x in valid_entries])
        return valid
    
    def get_id_and_num(self): # returns a dictionary of id and distance
        return {x['number']: x['distance'] for x in self.get_valid_entries()}
    


class Fit: # handles the fitting of curves to given data.
    def __init__(self):
        self.freq = []
        self.intensity = []
        self.x = []
        self.y = []
        self.error = 0


    def gaussian(self, freq, a, mu, sigma):
        return a*np.exp(-(freq-mu)**2/(2*sigma**2))

    def curve(self, freq, c, m, a, mu, sigma): # this is my curve function that I pass into scipy to be fitted to the data
        return c + freq*m + self.gaussian(freq, a, mu, sigma)


    def mu_guess(self): # guesses the mean of the data by finding the x value corresponding to the highest y value
        return self.x[np.where(self.y == np.amax(self.y))][0]


    def gradient_guess(self): # guesses the gradient of the graph based on an average y value of data near each end of the graph
        return (np.mean(self.y[-20:-1]) - np.mean(self.y[0:20])) / (self.x[-1] - self.x[0])
    

    def actually_fit(self, freq, intensity, num):

        self.freq = freq
        self.intensity = intensity
        self.x = freq
        self.y = intensity

        mu_guess1 = self.mu_guess()
        grad_guess = self.gradient_guess()
        line_fit = curve_fit(self.curve, self.freq, self.intensity, p0=[max(self.intensity), grad_guess, max(self.intensity)*10, mu_guess1, 0.4e14], maxfev = 120000)

        # line = self.line(self.freq, *line_fit[0])


        # print('-------------------------------------------')
        # print(f'results number {num}')
        # print(f'line = {line_fit[0][1]}x + {line_fit[0][0]}')
        # print(f'gaussian = {line_fit[0][2]}exp((x-{line_fit[0][3]})^2 / (2*{line_fit[0][4]}^2))')
        # print()
        self.error = line_fit[1]
        return line_fit[0][3] # this returns the 4th parameter in my scipy curve which is mu, the mean frequency of the guassian part
    

    def get_mu(self, freq, intensity, num):
        return self.actually_fit(freq, intensity, num)


def main():
    # from code before I cleaned the data, respective code commented in the function
    # freq, intensity = Data('Data/SpectralData_Hbeta.csv').process_data()

    valid_ids = Data('Data/dist_data.txt').get_valid_ids() # gets the valid observation numbers for the data
    x, y = Data('Data/SpectralData_Hbeta.csv').process_data() # gets the frequency and intensity data
    clean_x, clean_y, clean_x_ids, clean_y_ids = [], [], [], []
    
    for i in range(len(x)):
        if x[i][0] in valid_ids and y[i][0] == x[i][0]: # filters the x and y to only include valid observations
            clean_x.append(x[i][1:])
            clean_x_ids.append(x[i][0]) # makes a note of the valid observation numbers in the same order as the data
            clean_y.append(y[i][1:])
            clean_y_ids.append(y[i][0])

    clean_x = np.array(clean_x) # formats to a numpy array for easier manipulation
    clean_y = np.array(clean_y)
    freq, intensity = clean_x, clean_y

    # print(freq, intensity)

    fit = Fit()
    mus = []
    err = []
    for x in range(len(freq)): # iterates through the valid data and gets the valid mu values and associated covariance matrix
        mus.append([fit.get_mu(freq[x], intensity[x], x)])
        err.append(fit.error)
    # mus = np.array([fit.get_mu(freq[x], intensity[x], x) for x in range(len(freq))])
    mus = np.array(mus)
    mus_err = np.array([np.sqrt(x[3][3]) for x in err]) # gets the standard error on mu

    


    
        
    l_e = 486.1e-9
    c = 2.9979e8
    lamdas = c / mus

    # formula for the propogation of uncertainty from mu to lamda
    lamdas_err = np.array([lamdas[i] * np.abs(mus_err[i] / mus[i]) for i in range(len(mus))])


    vs = c * (((lamdas / l_e)**2 - 1) / (1 + (lamdas / l_e)**2))

    # propogates to velocities
    vs_err = np.array([np.abs((4*c*lamdas[i]/l_e) / ((1 + (lamdas[i] / l_e)**2))) * lamdas_err[i] for i in range(len(lamdas_err))])
    vs_err = np.squeeze(vs_err) / 1000 # converts the 0 dimensional arrays to singular values i.e. [[1], [2]] -> [1, 2]
    

    distance_dict = Data('Data/dist_data.txt').get_id_and_num() # gets a dictionary of observation number: distance
    
    # creates a dictionary of observation number: [x, y]
    graphs_dict = {clean_x_ids[i]: [distance_dict[clean_x_ids[i]], vs[i]] for i in range(len(vs))} 

    # gets the x and y axis for the final plot from that dictionary
    final_x = np.array([x[1][0] for x in graphs_dict.items()])
    final_y = np.squeeze(np.array([x[1][1] / 1000 for x in graphs_dict.items()]))


    # print(final_y)



    fig, ax = plt.subplots()
    ax.errorbar(final_x, final_y, yerr=(vs_err), fmt='o', capsize=5, label='') # plots velocities with error bars

    # fits a linear curve to the data with an inverse weighting on the range of the velocities
    line = np.polyfit(final_x, final_y, 1, w=1/vs_err, cov='unscaled')

    # plots that linear curve
    plt.plot(final_x, np.polyval(line[0], final_x), color="orange", label="Regression line")
    
    print(f'line = ({line[0][0]}+- {np.sqrt(line[1][0][0])}x  + ({line[0][1]}+-{np.sqrt(line[1][1][1])})')

    # Estimate of Hubble's constant: 71.01 +/- 1.95 km/s/Mpc
    
    ax.set_xlabel('Distance (Mpc)')
    ax.set_ylabel('Redshift (Km/s)')
    ax.set_title('Measuring the Hubble constant')
    ax.legend()
    ax.grid()
    plt.savefig('Out/graph.png')
    plt.show()



main()

