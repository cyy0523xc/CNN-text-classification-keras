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

    file_csv = './data/splitter_restore_mdr_comments.csv'
    file_pos = './data/normalized_uniform_positive.txt'
    file_neg = './data/normalized_uniform_negative.txt'
    normalizer = Normalizer()
    count_neg, count_pos = 0, 0

    with open(file_neg, 'w', encoding='utf-8') as nf, open(file_pos, 'w', encoding='utf-8') as pf:
        num_lines = 50000 #get_total_lines(file_csv)
        for row in tqdm(get_row(file_csv), total=num_lines):

            try:
                text, status = row[12], row[18]
                norm_text = normalizer.normalize(text)
            except IndexError:
                continue
            
            if not norm_text:
                continue
        
            if status in positive:
                file = pf                
                count_pos += 1
            else:                
                file = nf
                count_neg += 1
            
            file.write(norm_text+"\n")            

            if count_neg + count_pos >= num_lines:
                break
           

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
