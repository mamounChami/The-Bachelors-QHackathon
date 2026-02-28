import numpy as np

def generate_payment(x, int_threshold, prop):
    output = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] < 100_000_000 or x[i] > 500_000_000:
            output[i] = 0
        elif x[i] < int_threshold:
            output[i] = x[i]
        else:
            output[i] = prop * x[i]
    return output

def generate_profit(x, payment, pdf, premium):
    return np.sum((payment - x)*pdf)*(x[1] - x[0]) - premium

def generate_N(N, props, thresholds, premiums):
    loaded_pdf = np.load('kde_data.npz')
    x = loaded_pdf['x_final']
    y = loaded_pdf['y_final_per_year']
    r_values = np.zeros((1, N))
    for i in range(N):
        prop = props[i]
        int_threshold = thresholds[i]
        premium = premiums[i]
        
        payment = generate_payment(x, int_threshold, prop)
        profit = generate_profit(x, payment, y, premium)
        r_values[0, i] = profit
    
    return r_values
