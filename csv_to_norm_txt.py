import csv
from normalizer import Normalizer
from tqdm import tqdm


def get_total_lines(file):
    with open(file, encoding='utf-8') as f:
        reader = csv.reader(f)
        return sum(1 for row in reader) 


def get_row(file):
    with open(file, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            yield row


if __name__ == '__main__':

    # statuses in csv
    positive = [
        'active',
        'auto_activated'
    ]

    negative = [
        'locked',
        'locked_by_complaint',
        'locked_by_complaint_auto',
        'locked_by_word'
    ]

    file_csv = './data/100_000_bad_good.csv'
    file_pos = './data/normalized_positive.txt'
    file_neg = './data/normalized_negative.txt'
    normalizer = Normalizer()
    count_neg, count_pos = 0, 0

    with open(file_neg, 'w', encoding='utf-8') as nf, open(file_pos, 'w', encoding='utf-8') as pf:
        num_lines = get_total_lines(file_csv)
        for text, status in tqdm(get_row(file_csv), total=num_lines):
            norm_text = normalizer.normalize(text)
            if not norm_text:
                continue

            if status in positive:
                file = pf
                count_pos += 1
            else:
                file = nf
                count_neg += 1
            
            file.write(norm_text+"\n")
           

    print("""Split {csv} as normalized
        positive {pos} - {count_pos} samples
        negative {neg} - {count_neg} samples
    """.format(
        csv=file_csv,
        pos=file_pos,
        neg=file_neg,
        count_pos=count_pos,
        count_neg=count_neg)
    )
