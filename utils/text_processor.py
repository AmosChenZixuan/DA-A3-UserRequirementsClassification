import contractions
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class TextProcessor:

    @staticmethod
    def preprocess_pipeline(df, col, stopwords):
        tp = TextProcessor
        df = df.copy() # to avoid some warnings (SettingWithCopyWarning)
        # is, ASAP !!!!
        df = tp.apply_contractions(df, col)
        # is, AS SOON AS POSSIBLE !!!!
        df = tp.remove_nonAlphanumeric(df, col)
        # is  AS SOON AS POSSIBLE
        df = tp.to_lower(df, col)
        # is  as soon as possible
        df = tp.apply_lemmatization(df, col)
        # be  as soon as possible
        if not stopwords:
            df = tp.remove_stopwords(df, col)
        #        soon    possible
        df = tp.remove_extra_spaces(df, col)
        # soon possible
        return df


    @staticmethod
    def apply_contractions(df, col):
        df[col] = df[col].apply(contractions.fix)
        return df

    @staticmethod
    def remove_nonAlphanumeric(df, col):
        df[col] = df[col].str.replace(r"[^a-zA-Z]", " ", regex=True)
        return df

    @staticmethod
    def to_lower(df, col):
        df[col] = df[col].str.lower()
        return df

    @staticmethod
    def apply_lemmatization(df, col):
        wordnet_lemmatizer = WordNetLemmatizer()

        def lemmatize(field):
            tokens = field.split()
            lemmatized = [wordnet_lemmatizer.lemmatize(word, pos="v") for word in tokens]
            return " ".join(lemmatized)

        df[col] = df[col].apply(lemmatize)
        return df

    @staticmethod
    def remove_stopwords(df, col):
        stop_words = list(stopwords.words('english'))
        for stop_word in stop_words:
            pattern = r"\b" + stop_word + r"\b"
            df[col] = df[col].str.replace(pattern, '', regex=True)

        return df 


    @staticmethod
    def remove_extra_spaces(df, col): 
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        return df