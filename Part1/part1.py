#!/usr/bin/env python3

import numpy as np
import pandas as pd


def generate_data():
    np.random.seed(0)

    col1 = np.random.normal(loc=0, scale=1, size=300)  # mean=0, std=1
    col2 = np.random.normal(loc=5, scale=2, size=300)  # mean=5, std=2
    col3 = np.random.randint(low=10, high=20, size=300)  # integers
    col4 = np.random.normal(loc=10, scale=0.5, size=300)  # floats
    col5 = np.random.normal(loc=2.5, scale=0.1, size=300)  # mean close to 2.5

    noise = np.random.normal(size=300)
    col6 = col1 + 2 * noise  # positive correlation with col1
    col7 = -col2 + 3 * noise  # negative correlation with col2
    col8 = np.random.normal(loc=0, scale=1, size=300)  # zero correlation with others

    df = pd.DataFrame({
        'col1': col1,
        'col2': col2,
        'col3': col3,
        'col4': col4,
        'col5': col5,
        'col6': col6,
        'col7': col7,
        'col8': col8
    })

    df.to_csv('artificial_dataset.csv', index=False)


if __name__ == '__main__':
    generate_data()
