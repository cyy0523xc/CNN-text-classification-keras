import csv
import click
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from data_helpers import load_data, MAX_LENGTH
from normalizer import Normalizer


POSITIVE = [
    'active',
    'auto_activated'
]

SKIP = [
    'locked_by_ip',
    'unapproved'
]


def get_total_lines(file):
    with open(file, encoding='utf-8') as f:
        reader = csv.reader(f)
        return sum(1 for row in reader) 


def fetch_samples(file, skip=0, limit=None):    
    i = 0
    with open(file, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:

            if i < skip:
                i += 1
                continue

            if limit and i > skip + limit:
                break

            try:
                text, status = row[12], row[18]
                i += 1
                yield text, status
            except IndexError as e:
                # print('Warning: cant find status or text')
                continue            


def get_vector(text, vocabulary, pad_token):
    sequence = []
    unknown_words_count = 0
    for word in text.split(' '):
        try:
            token = vocabulary[word]
        except KeyError as e:
            # click.echo(click.style('Warning: "{}" not in vocabulary, replaced with padding'.format(word), fg='yellow'))
            unknown_words_count += 1
            token = pad_token

        sequence.append(token)

    text_x = pad_sequences([sequence], padding='post', maxlen=MAX_LENGTH, value=pad_token)
    return text_x, unknown_words_count


def predict_status(pos_score, active_threshold, locked_threshold):
    if pos_score > active_threshold:
        return True
    elif locked_threshold <= pos_score <= active_threshold:
        return None
    else:
        return False


def is_correct(predict_status, actual_status):
    if predict_status is None:
        return ''
    elif predict_status == actual_status:
        return 1
    else:
        return 0


@click.command()
@click.argument('model_file', type=click.Path(exists=True))
@click.argument('data_file', type=click.Path(exists=True))
@click.argument('stat_file', type=click.Path())
def evaluate(model_file, data_file, stat_file):

    max_lines = 100000
    normalizer = Normalizer()
    x, y, vocabulary, vocabulary_inv = load_data(normalize=False)
    pad_token = vocabulary['<PAD/>']
    model = load_model(model_file)

    # click.echo(click.style("Calculating lines count...", fg="yellow"))
    # total_lines = get_total_lines(data_file)

    csv_header = [
        'comment', 
        'norm.comment', 
        'unknown words', 
        'positive score', 
        'negative score', 
        'actual status', 
        'predicted active', 
        'not sure',
        'predicted locked',
        'correct',
    ]

    sf = open(stat_file, "w", encoding='utf-8')
    writer = csv.writer(sf, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(csv_header)

    samples = fetch_samples(data_file, skip=100000, limit=max_lines)
    for comment, actual_status in tqdm(samples, total=max_lines):
        if actual_status in SKIP:
            continue

        normalized = normalizer.normalize(comment)        
        text_x, unknown_words_count = get_vector(normalized, vocabulary, pad_token)
        prediction = model.predict(text_x)
        neg_score, pos_score = prediction[0][0], prediction[0][1]
        
        bool_status = actual_status in POSITIVE        
        predicted_status = predict_status(pos_score, active_threshold=0.67, locked_threshold=0.30)

        writer.writerow([
            comment,
            normalized,
            unknown_words_count,
            pos_score,
            neg_score,
            actual_status,            
            int(predicted_status == True),
            int(predicted_status is None),
            int(predicted_status == False),
            is_correct(predicted_status, bool_status)
        ])


if __name__ == '__main__':

    evaluate()