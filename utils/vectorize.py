from sklearn.feature_extraction.text import TfidfVectorizer


class Vectorizer:

    @staticmethod
    def tfidf(train, test, ngram_range=(1,2), \
                            min_df=0.01, max_df=0.8, \
                            max_features=100):

        tfidf = TfidfVectorizer(encoding='utf-8',ngram_range=ngram_range,
                                stop_words=None,lowercase=False,
                                max_df=max_df,
                                min_df=min_df,
                                max_features=max_features,
                                norm='l2',
                                sublinear_tf=True)

        features_train = tfidf.fit_transform(train).toarray()
        features_test = tfidf.transform(test).toarray()
        return features_train, features_test
