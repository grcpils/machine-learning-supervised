import numpy as np
import pandas as pd

WEIGHTS = {'age': 1, 'height': 1, 'job': 2, 'city': 12, 'favorite music style': 5}


def custom_metric(row1, row2):
    distance_age = row1['age'] - row2['age']
    distance_height = row1['height'] - row2['height']
    distance_job = 1 if row1['job'] == row2['job'] else 0
    distance_city = 1 if row1['city'] == row2['city'] else 0
    distance_music = 1 if row1['favorite music style'] == row2['favorite music style'] else 0

    return np.sqrt(WEIGHTS['age'] * distance_age ** 2 + WEIGHTS['height'] * distance_height ** 2 + WEIGHTS[
        'job'] * distance_job ** 2 + WEIGHTS['city'] * distance_city ** 2 + WEIGHTS[
                       'favorite music style'] * distance_music ** 2)


if __name__ == '__main__':
    df = pd.read_csv('dataset.csv')
    dissimilarity_matrix = np.zeros((len(df), len(df)))
    for i in range(len(df)):
        for j in range(len(df)):
            dissimilarity_matrix[i, j] = custom_metric(df.iloc[i], df.iloc[j])

    mean_dissimilarity = np.mean(dissimilarity_matrix)
    std_dissimilarity = np.std(dissimilarity_matrix)

    print('Mean dissimilarity:', mean_dissimilarity)
    print('Standard deviation of dissimilarity:', std_dissimilarity)

    np.save('dissimilarity_matrix.npy', dissimilarity_matrix)
