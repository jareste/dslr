import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import parse_csv, is_float

def main(file_name):
    parsed_data = parse_csv(file_name)
    headers = list(parsed_data[0])
    data = parsed_data[1:]
    
    house_index = headers.index('Hogwarts House')
    houses = [row[house_index] for row in data]
    numerical_columns = [i for i, value in enumerate(data[0]) if is_float(value)]
    
    numerical_data = {}
    for i in numerical_columns:
        col_name = headers[i]
        if col_name == 'Index':
            continue
        numerical_data[col_name] = [float(row[i]) if is_float(row[i]) else np.nan for row in data]
    
    numerical_data['Hogwarts House'] = houses

    df = pd.DataFrame(numerical_data)

    sns.pairplot(df, hue='Hogwarts House', palette={'Gryffindor': 'red', 'Hufflepuff': 'yellow', 'Ravenclaw': 'blue', 'Slytherin': 'green'}, diag_kind='hist')
    
    plt.savefig('/output/pair_plot.png', dpi=500, format='png', bbox_inches='tight', pad_inches=0.1, transparent=False)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pair_plot.py <dataset.csv>")
    else:
        file_name = sys.argv[1]
        main(file_name)
