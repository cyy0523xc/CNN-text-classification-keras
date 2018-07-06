import click
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from data_helpers import load_data


@click.command()
@click.option('--neg-file', help='负面情感语料文件')
@click.option('--pos-file', help='正面情感语料文件')
@click.option('--checkpoint', default=None, help='Continue training from checkpoint.')
@click.option('--epoch', default=20, help='Initial epoch number.')
def train(neg_file, pos_file, checkpoint, epoch):
    """
    Examples:
        python3 model.py --neg-file=/var/www/src/github.com/nlp/ChineseNlpCorpus/format_datasets/total_train_neg.txt --pos=/var/www/src/github.com/nlp/ChineseNlpCorpus/format_datasets/total_train_pos.txt
    """
    click.echo(click.style('Loading data...', fg='green'))
    x, y, vocabulary, vocabulary_inv = load_data(pos_file, neg_file)
    click.echo(click.style('Loading data over.', fg='green'))

    # x.shape -> (10662, 56)
    # y.shape -> (10662, 2)
    # len(vocabulary) -> 18765
    # len(vocabulary_inv) -> 18765

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # X_train.shape -> (8529, 56)
    # y_train.shape -> (8529, 2)
    # X_test.shape -> (2133, 56)
    # y_test.shape -> (2133, 2)


    sequence_length = x.shape[1] # 56
    vocabulary_size = len(vocabulary_inv) # 18765
    embedding_dim = 256
    filter_sizes = [3,4,5]
    num_filters = 512
    drop = 0.5

    epochs = 100
    batch_size = 500 # initial 30

    if checkpoint:
        click.echo(click.style('Loading model %s...' % checkpoint, fg='green'))
        model = load_model(checkpoint)
    else:
        # this returns a tensor
        click.echo(click.style('Creating new model...', fg='green'))
        inputs = Input(shape=(sequence_length,), dtype='int32')
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
        reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

        maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(drop)(flatten)
        output = Dense(units=2, activation='softmax')(dropout)

        # this creates a model that includes
        model = Model(inputs=inputs, outputs=output)
        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    # tensorboard callback
    cb_tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    cb_checkpoint = ModelCheckpoint('./checkpoints/model.epoch.{epoch:03d}.vacc{val_acc:.4f}.hdf5',
                                    monitor='val_acc', verbose=1, save_weights_only=False, save_best_only=True, mode='auto')

    click.echo(click.style('Traning model...', fg='green'))

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, initial_epoch=epoch,
        verbose=1, callbacks=[cb_checkpoint, cb_tensorboard], validation_data=(X_test, y_test)
    )


if __name__ == '__main__':
    train()
