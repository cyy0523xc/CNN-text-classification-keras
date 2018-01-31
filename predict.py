import click
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from data_helpers import load_data, MAX_LENGTH
from normalizer import Normalizer
import json

@click.command()
@click.argument('model_file', type=click.Path(exists=True))
@click.argument('data_file', type=click.Path(exists=True))
def predict(model_file, data_file):
    normalizer = Normalizer()

    click.echo(click.style("Loading data, building vocabulary...", fg="yellow"))
    x, y, vocabulary, vocabulary_inv = load_data(normalize=False)
    pad_token = vocabulary['<PAD/>']

    click.echo(click.style("Loading model...", fg="yellow"))
    model = load_model(model_file)

    samples = list(open(data_file, "r", encoding='utf-8').readlines())

    for sample in samples:
        click.echo(click.style("Input: %s" % sample.strip(), fg="yellow"))
        sample = normalizer.normalize(sample)
        click.echo(click.style("Normalized: %s" % sample, fg="yellow"))

        sequence = []
        for word in sample.split(' '):
            try:
                token = vocabulary[word]
            except KeyError as e:
                click.echo(click.style('Warning: "{}" not in vocabulary, replaced with padding'.format(word), fg='yellow'))
                token = pad_token

            sequence.append(token)

        text_x = pad_sequences([sequence], padding='post', maxlen=MAX_LENGTH, value=pad_token)

        prediction = model.predict(text_x)
        neg_score, pos_score = prediction[0][0], prediction[0][1]

        # click.echo(click.style("Vector: %s" % sequence, fg="yellow"))
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
