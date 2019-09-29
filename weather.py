from main import FFNeuralNetwork
from scipy.io import arff
import pandas as pd

# Read dataset
data = arff.loadarff('data/weather.arff')
df = pd.DataFrame(data[0])

## DATA PREPROCESSING
# Convert categorical variable to numerical variable
def convert_categorical_to_numerical(df, column_name):
    df[column_name] = df[column_name].astype('category')
    categories = df.select_dtypes(['category']).columns
    df[categories] = df[categories].apply(lambda x: x.cat.codes)
    return df

df = convert_categorical_to_numerical(df, 'outlook')
df = convert_categorical_to_numerical(df, 'windy')
df = convert_categorical_to_numerical(df, 'play')

# DATA TRAINING
dataset = df.as_matrix()

# breakpoint buat debug2
# import code; code.interact(local=dict(globals(), **locals()))

nb_hidden_layer = 1
nb_nodes = 4
nb_input = dataset.shape[1] - 1
model = FFNeuralNetwork(nb_hidden_layer, nb_nodes, nb_input)

epoch = 20
l_rate = 0.001
momentum = 0.25
batch_size = 2
model.fit(dataset, epoch, l_rate, momentum, batch_size)
print(model.predict(dataset[:,:-1]))
