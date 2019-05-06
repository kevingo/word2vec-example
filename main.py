from gensim.models import word2vec

params = {
    'model': './model/word2vec_3.model'
}

def main():
    model = word2vec.Word2Vec.load(params['model'])

    # show two terms similarity
    sim = model.similarity('man', 'woman')
    print(sim)

    # compute cosine similarity between two sets of words.
    n_sim = model.n_similarity(['sushi', 'shop'], ['japanese', 'restaurant'])
    print(n_sim)

    

if __name__ == "__main__":
    main()