import gzip
import gensim
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

param = {
    'input_file': './data/reviews_data.txt.gz',
    'epoch': 3,
    'size': 150, # num of dim on the output vector
    'window': 10, # The maximum distance between the target word and its neighboring word
    'min_count': 2, # Minimium frequency count of words. The model would ignore words that do not satisfy the min_count
    'worker': 4
}

def read_data(input_file):
    logging.info("reading file {0} ".format(input_file))
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate (f):
            # print log for reading 10000 reviews
            if (i % 10000 == 0):
                logging.info("read {0} reviews".format(i))

            # gensim simple preprocess
            # https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess
            # 1. Convert a document into a list of lowercase tokens,
            # 2. Ignore tokens that are too short or too long.
            yield gensim.utils.simple_preprocess(line)

def train_word2vec(train_data):
    model = gensim.models.Word2Vec(
        train_data,
        size=param['size'],
        window=param['window'],
        min_count=param['min_count'],
        workers=param['worker'])
    model.train(train_data, total_examples=len(train_data), epochs=param['epoch'])
    return model

if __name__ == "__main__":
    print('Word2Vec training.')
    try:
        documents = list(read_data(param['input_file']))
        model = train_word2vec(documents)
        path = './model/word2vec_' + str(param['epoch']) + '.model'
        model.save(path)
    except Exception as e:
        logging.info('train word2vec error: ', e)

