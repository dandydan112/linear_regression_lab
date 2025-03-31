import numpy as np

def load_data():
    x_train = np.array([
        6.1101, 5.5277, 8.5186, 7.0032, 5.8598,
        8.3829, 7.4764, 8.5781, 6.4862, 5.0546,
        # You can add more if needed
    ])
    y_train = np.array([
        17.592, 9.1302, 13.662, 11.854, 6.8233,
        11.886, 4.3483, 12.0, 6.5987, 3.8166,
        # You can add more if needed
    ])
    return x_train, y_train
