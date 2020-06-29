import string
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from collections import Counter
import math
import operator
from itertools import repeat, chain
import matplotlib.pyplot as plt

##############################################################################################################
# -------------------------------------------------- TASK 1 -------------------------------------------------#
##############################################################################################################

csv = pd.read_csv('hns_2018_2019.csv')  # reading csv with pandas
cvs_values = [v for v in csv.values]
DELTA = 0.5

# training and tests sets
training_set, test_set, word_tokens = [], [], []


def get_column_idx(columns):
    for i, column in enumerate(columns):
        if re.search('Title', column, re.IGNORECASE):
            idx_title = i
        elif re.search('Post Type', column, re.IGNORECASE):
            idx_post_type = i
        elif re.search('Created At', column, re.IGNORECASE):
            idx_created_at = i

    return idx_title, idx_post_type, idx_created_at


def tokenizer(text):
    text = re.sub(r"[^\w'\w’\s\&\+-]+", '', text.lower()) \
        .replace("""\'s""", '') \
        .replace('’s', '') \
        .replace("""\'""", '')
    return text


# Obtaining the index of the following 3 columns: [ Title | Post Type | Created At ]
idx_title, idx_post_type, idx_created_at = get_column_idx(csv.columns.values)

# Extracting data for tokenizing words and for training and testing sets
for row in cvs_values:
    if re.search('2018', row[idx_created_at]):
        training_set.append(row)
        title_class_cell = [tokenizer(row[idx_title]), row[idx_post_type]]
        for cell in title_class_cell[0].split():
            word_class_cell = [cell, title_class_cell[1]]
            word_tokens.append(word_class_cell)

    elif re.search('2019', row[idx_created_at]):
        test_set.append(row)

# splitting list sentences into a 2d list of words with their associated classes
word_tokens = [[re.sub(r"^\W+|\W+$", "", w[0]), w[1]] for w in word_tokens]  # removing non-alphanumeric characters

# extracting and storing useless words and removing them tokens list
useless_words = sorted(list(set([x[0] for x in word_tokens if (x[0].isdecimal() or len(x[0]) <= 1)])))
word_tokens = sorted([x for x in word_tokens if not (x[0].isdecimal() or len(x[0]) <= 1)], key=lambda x: x[0])
word_dictionary = {'word': [w[0] for w in word_tokens],
                   'post_type': [w[1] for w in word_tokens]}
vocabulary = sorted(list(set(w[0] for w in word_tokens)))  # storing unique words into vocabulary list
words_per_class = Counter([w[1] for w in word_tokens])  # class frequency for each word
word_frequency = Counter([w[0] for w in word_tokens])

# writing vocabulary list to vocabulary.txt
with open('vocabulary.txt', 'a+') as vocab_file:
    vocab_file.writelines("%s\n" % v for v in vocabulary)  # writing to vocabulary.txt

# creating remove_word.txt for storing removed words
with open('remove_word.txt', 'a+') as removed_file:
    removed_file.writelines("%s\n" % u for u in useless_words)  # writing to remove_word.txt

df_word_type = pd.DataFrame(word_dictionary, columns=word_dictionary.keys())
frequency = df_word_type.groupby(['word', 'post_type']).size()


def write_to_file(data_list, file_name):
    with open(file_name, 'w+') as f:
        for l, i in enumerate(data_list, start=1):
            f.write(f'{l}  ')
            for j in i:
                f.write(f'{j}  ')
            f.write('\n')


# training model
def build_model(vocab, data_set, file_name):
    class_order = ['story', 'ask_hn', 'show_hn', 'poll']
    train_classes = [cl for s in class_order for cl in list(set([x[idx_post_type] for x in data_set])) if cl == s]
    model_dict = {}
    model_to_txt = []

    for word in vocab:
        table_row = [word]
        for cl in train_classes:

            # noinspection PyBroadException
            try:
                fq = frequency[(word, cl)]
            except Exception:
                fq = 0

            pr = round((fq + DELTA) / (words_per_class[cl] + (len(vocab) * DELTA)), 7)
            model_dict[(word, cl)] = {'frequency': fq, 'probability': pr}

            table_row.extend([fq, pr])
        model_to_txt.append(table_row)
    write_to_file(model_to_txt, file_name)

    return model_to_txt


model_dictionary = build_model(vocabulary, training_set, 'model.txt')


##############################################################################################################
# -------------------------------------------------- TASK 2 -------------------------------------------------#
##############################################################################################################

# classifier for test dataset
def classify(data_set, vocab, model, file_name=None):
    class_order = ['story', 'ask_hn', 'show_hn', 'poll']
    posts = [[d[idx_title], d[idx_post_type]] for d in data_set]
    test_classes = [cl for s in class_order for cl in list(set([x[idx_post_type] for x in data_set])) if cl == s]
    class_count_dict = Counter([w[idx_post_type] for w in data_set])
    total_class_sample = len([cl[idx_post_type] for cl in data_set])
    guesses = []
    result_to_file = []

    for item in posts:
        scores_order = []
        scores = {}
        words = [w.strip() for w in item[0].split(' ')]
        result_tf_row = [item[0]]
        for cl in test_classes:
            scores[cl] = math.log10(class_count_dict[cl] / total_class_sample)
            for word in words:
                # noinspection PyBroadException
                try:
                    scores[cl] += math.log10(model[(word, cl)]['probability'])
                except Exception:
                    scores[cl] += math.log10(DELTA / (words_per_class[cl] + (len(vocab) * DELTA)))

            scores_order.append(scores[cl])
        classification = max(scores.items(), key=operator.itemgetter(1))[0]
        result_tf_row.extend([classification])

        for s in scores_order:
            result_tf_row.extend([s])

        result = 'right' if classification == item[1] else 'wrong'
        result_tf_row.extend([item[1], result])
        result_to_file.append(result_tf_row)

    if file_name is not None:
        write_to_file(result_to_file, file_name)

    return guesses


result = classify(test_set, vocabulary, model_dictionary, 'baseline-result.txt')

##############################################################################################################
# -------------------------------------------------- TASK 3 -------------------------------------------------#
##############################################################################################################

# ------- Experiment 1 -------

with open('stopwords.txt') as f:
    stop_words_list = [line.rstrip('\n') for line in f]


def stop_word_filter(vocab, stop_words):
    filtered = [w for w in vocab if w not in stop_words]

    return filtered


stop_word_vocab = stop_word_filter(vocabulary, stop_words_list)

stop_word_model = build_model(stop_word_vocab, training_set, 'stopword-model.txt')
accuracy_exp1 = classify(test_set, stop_word_vocab, stop_word_model, 'stopword-result.txt')

# ------- Experiment 2 -------

word_len_vocab = [v for v in vocabulary if not (len(v) <= 2 or len(v) >= 9)]
word_len_model = build_model(word_len_vocab, training_set, 'wordlength-model.txt')
accuracy_exp2 = classify(test_set, word_len_vocab, word_len_model, 'wordlenth-result.txt')

# ------- Experiment 3 -------

word_lst = [w[0] for w in word_tokens]
desc_ordered_words = list(chain.from_iterable(repeat(i, c)
                                              for i, c in Counter(word_lst).most_common()))  # order words by frequency
desc_vocab = list(dict.fromkeys(desc_ordered_words))
word_count_dict = Counter(desc_ordered_words)

accuracy_freq = []
remaining_words_freq = []

conditions = ['(val == 1)', '(val <= 5)', '(val <= 10)', '(val <= 15)', '(val <= 20)']

for c in conditions:
    filtered_freq_vocab = list({key: val for key, val in word_count_dict.items() if not bool(c)})
    filtered_model = build_model(filtered_freq_vocab, training_set, 'filterfreq-model.txt')
    filtered_acc = Counter(classify(test_set, filtered_freq_vocab, filtered_model, 'filteredfrep-results.txt'))
    accuracy_freq.append(filtered_acc['right'])
    remaining_words_freq.append(len(vocabulary) - len(filtered_freq_vocab))

accuracy_percent = []
remaining_words_percent = []
percentage = [5, 10, 15, 20, 25]

for p in percentage:
    percentage_idx = int(len(desc_vocab) * (100 - p) / 100)
    filter_percent_vocab = [j for i, j in enumerate(desc_vocab) if not (i <= percentage_idx)]
    filter_percent_model = build_model(filter_percent_vocab, training_set, 'filterpercent-model.txt')
    filter_percent_acc = Counter(classify(test_set, filter_percent_vocab,
                                          filter_percent_model, 'filterpercent-results.txt'))
    accuracy_percent.append(filter_percent_acc['right'])
    remaining_words_percent.append(len(vocabulary) - len(filter_percent_vocab))

plt.plot(accuracy_freq, remaining_words_freq, color='green', linestyle='dashed', linewidth=3,
         marker='o', markerfacecolor='blue', markersize=12)
plt.xlim(1, 8)
plt.xlim(1, 8)

plt.xlabel('Correct guesses')
plt.xlabel('Number of remaining words')

plt.title('Frequency Word Filtering')

plt.show()

plt.plot(accuracy_percent, remaining_words_percent, color='green', linestyle='dashed', linewidth=3,
         marker='o', markerfacecolor='blue', markersize=12)
plt.xlim(1, 8)
plt.xlim(1, 8)

plt.xlabel('Correct guesses')
plt.xlabel('Number of remaining words')

plt.title('Percent Frequency Word Filtering')

plt.show()
