from cGAN import *
from utils import *
# from utils_new import *
import time

if __name__ == '__main__':
    t_start = time.time()
    # Prepare data
    df_train = pd.read_csv('new_train_all.csv')
    df_train['Label'] = df_train['Label'].astype(np.int64)

    # df_train.drop(columns='Class', inplace=True)
    # Eliminate extreme cases
    df_train_new = df_train[(df_train.iloc[:, :-2] >= 0).all(axis=1)]
    df_train_new = df_train_new[(df_train_new.iloc[:, :-2] <= 100).all(axis=1)]

    # Normalize
    max_value = df_train_new.iloc[:, :-2].max().max()
    min_value = df_train_new.iloc[:, :-2].min().min()
    scale = max_value - min_value

    df_train_new_norm = (df_train_new.iloc[:, :-2] - min_value)/scale
    df_train_new_norm['Label'] = df_train_new['Label']

    data_train = torch.tensor(df_train_new_norm.to_numpy())

    # Set parameters
    input_size = 24
    hidden_size = 64
    label_size = len(df_train.Label.unique())
    output_size = 24

    lr = 0.0001
    epochs = 10000
    batch_size = 2000

    num_samples = 2000
    folder_name = 'cGAN generation'

    # Initialize the cGAN
    gan = cGAN(input_size, label_size, hidden_size, output_size, lr)
    # Train the GAN
    gan.train(data_train, epochs, batch_size)

    # Generated data
    real_results(df_train_new, folder_name)
    generate_results(gan, num_samples, folder_name, min_value, scale)

    stats_plot(folder_name)

    print(f'Time: {round((time.time() - t_start)/60, 2)} minutes')
