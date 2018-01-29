import csv
from normalizer import Normalizer
from tqdm import tqdm


def get_total_lines(file):
    with open(file, encoding='utf-8') as f:
        reader = csv.reader(f)
        return sum(1 for row in reader) 


def fetch_from_csv(file):
    with open(file, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            yield row[0]


if __name__ == '__main__':

    in_file_name = './data/comments_bad.csv'
    out_file_name = './data/comments_bad_norm.txt'
    normalizer = Normalizer()

    with open(out_file_name, 'w', encoding='utf-8') as out_file:
        num_lines = get_total_lines(in_file_name)
        for line in tqdm(fetch_from_csv(in_file_name), total=num_lines):
            out_file.write(normalizer.normalize(line) + "\n")

    print("Saved {} as normalized {}".format(in_file_name, out_file_name))