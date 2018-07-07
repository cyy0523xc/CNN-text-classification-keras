import click
import jieba
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from data_helpers import load_data
#from data_helpers import clean_str


@click.command()
@click.option('--checkpoint', default=None, help='Continue training from checkpoint.')
@click.option('--data-file', type=click.Path(exists=True))
@click.option('--neg-file', type=str)
@click.option('--pos-file', type=str)
def predict(checkpoint, data_file, neg_file, pos_file):
    """
    python predict.py weights.003-0.6235.hdf5 test_data.txt
    python predict.py weights.040-0.7604.hdf5 test_data.txt
    """
    click.echo(click.style("Loading data...", fg="yellow"))
    x, y, vocabulary, vocabulary_inv = load_data(pos_file, neg_file)
    pad_token = vocabulary['<PAD/>']

    click.echo(click.style("Loading model...", fg="yellow"))
    model = load_model(checkpoint)

    samples = list(open(data_file, encoding='utf8').readlines())

    for sample in samples:
        #sample = clean_str(sample)
        sequence = [vocabulary[word] for word in jieba.lcut(sample)]
        text_x = pad_sequences([sequence], padding='post', maxlen=56, value=pad_token)

        prediction = model.predict(text_x)
        neg_score, pos_score = prediction[0][0], prediction[0][1]

        # click.echo(click.style("Input: %s" % sequence, fg="yellow"))
        click.echo(click.style("Input: %s" % sample, fg="yellow"))
        click.echo(click.style("Positive: %s" % pos_score, fg="green"))
        click.echo(click.style("Negative: %s" % neg_score, fg="red"))
        print("\n")

    # tk = Tokenizer(
    #         num_words=2000,
    #         filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    #         lower=True,
    #         split=" ")

    # tk.fit_on_texts(text)
    # sequence = tk.texts_to_sequences(text)
    # seq_length = len(sequence[0])
    # x = np.array([sequence])

    #print(load_data())

if __name__ == '__main__':
    predict()
