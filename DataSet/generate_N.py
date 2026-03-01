import numpy as np
import os

def generate_payment(int_threshold, prop):
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'kde_data.npz')
    loaded_pdf = np.load(file_path)
    x = loaded_pdf['x_final']
    pdf = loaded_pdf['y_final_per_year']

    output = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] < 100_000_000 or x[i] > 500_000_000:
            output[i] = 0
        elif x[i] < int_threshold:
            output[i] = x[i]
        else:
            output[i] = prop * x[i]
    return np.sum(output * pdf) * (x[1] - x[0])

def expected_loss():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'kde_data.npz')
    loaded_pdf = np.load(file_path)
    x = loaded_pdf['x_final']
    y = loaded_pdf['y_final_per_year']
    return np.sum(x * y) * (x[1] - x[0])

def generate_N(N, props, thresholds, premiums):

    r_values = np.zeros((1, N))
    for i in range(N):
        prop = props[i]
        int_threshold = thresholds[i]
        premium = premiums[i]
        
        payment = generate_payment(int_threshold, prop)
        r_values[0, i] = payment - premium
    
    return r_values
