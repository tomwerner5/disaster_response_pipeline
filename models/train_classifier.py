import sys
import numpy as np
import pandas as pd
import re

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
#from nltk.corpus import stopwords as nltk_sw
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.metrics import (classification_report,
                             roc_auc_score, accuracy_score,
                             make_scorer)
from sklearn.ensemble import (RandomForestClassifier
                              ,GradientBoostingClassifier
                              ,AdaBoostClassifier
                              )
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
try:
    from sklearn.externals import joblib
except (ImportError, ModuleNotFoundError):
    import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sqlalchemy import create_engine


def load_data(database_filepath):
    '''
    Loads data from the sqlite database (DisasterResponse.db).

    Parameters
    ----------
    database_filepath : str
        The filepath for the database. Note that this is also assumed to be the
        table name.

    Returns
    -------
    X : pandas dataframe
        The design matrix (disaster responses).
    Y : pandas dataframe
        The response values (0, 1) corresponding to categories.
    category_names : list
        A list of the category names.

    '''
    # Create sqlite engine
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(database_filepath, con=engine)
    
    # List of features present
    feature_list = ['id', 'message', 'original', 'genre']
    
    # Full X and Y
    X = df[feature_list]
    Y = df.drop(feature_list, axis=1)
    
    # Reduce X to text column and series
    X = X['message']
    category_names = list(Y.columns)
    return X, Y, category_names


def tokenize(text, stem_type='lemma'):
    '''
    Preprocess a string for inclusion in a model downstream. 
    
    Process includes (in order):
        - Normalizing
        - Tokenizing
        - Removing Stopwords
        - Stemming/Lemmatization

    Parameters
    ----------
    text : str
        A string for processing.
    stem_type : str, optional
        Whether to use stemming ('stem') or lemmatization ('lemma').
        The default is 'lemma'.

    Raises
    ------
    ValueError
        Raised if stemmer choice is invalid.

    Returns
    -------
    stem_tokens : list
        A list of the processed text as tokens.

    '''
    ## Normalize text
    
    # Remove urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_regex, " ", text)
    
    # Remove unneeded characters
    # 
    # Uncomment the long regex to control special characters more specifically.
    # Originally, I had considered "'" to be an important character to keep,
    # but in the final modeling, it didn't seem to make much of a difference,
    # so the expression below is much simpler
    text = text.lower()
    #text = re.sub("([0-9\.\,\:\;\(\)\"\@\[\]\#\%\*\=\>\<\^\!\~\`\$\_\{\}\|\?\-\/\&\+\\\])", " ", text)
    text = re.sub("[^0-9a-z]", " ", text)
    
    ## Tokenize
    tokens = word_tokenize(text)
    
    ## Remove Stopwords
    
    # Note: The below stopwords are from the nltk english set. However, due to
    # pickling issues, I had to write them out in order to have consistent 
    # behavior. Ideally, one could remove the long list below and simply import
    # the list as-is using the commented line below, but since this seems to
    # be a known pickling issue, left the stopwords list written out, 
    #       see below for the issue:
    # https://stackoverflow.com/questions/44911539/pickle-picklingerror-args0-from-newobj-args-has-the-wrong-class-with-hado
    #
    #stopwords = nltk_sw.words('english')
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
                 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
                 'its', 'itself', 'they', 'them', 'their', 'theirs',
                 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                 "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
                 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
                 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
                 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
                 'for', 'with', 'about', 'against', 'between', 'into',
                 'through', 'during', 'before', 'after', 'above', 'below','to',
                 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                 'under', 'again', 'further', 'then', 'once', 'here', 'there',
                 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
                 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
                 "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
                 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                 "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
                 "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
                 "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
                 "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
                 "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    
    tokens = [tok for tok in tokens if tok not in stopwords]
    
    ## Stem/Lemmatization
    stem_tokens = []
    if stem_type is not None:
        if stem_type == 'stem':
            stemmer = SnowballStemmer("english")
            stem_tokens = [stemmer.stem(tok) for tok in tokens]
        elif stem_type == 'lemma':
            stemmer = WordNetLemmatizer()
            stem_tokens = [stemmer.lemmatize(tok) for tok in tokens]
        else:
            raise ValueError("Not a valid stemmer choice")
    else:
        print("No stemmer used")
        stem_tokens = tokens
    
    return stem_tokens


def build_model():
    '''
    Convenience function for building the model pipeline and performing
    a grid search on that pipeline. This function returns the model before 
    performing any fitting

    Returns
    -------
    model : sklearn GridSearchCV object
        The model object.

    '''
    # The pipeline was tested with a few different models. Uncomment the
    # respective models if desired.
    pipeline = Pipeline([
        ('count_vec', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        #('multi_output_clf', MultiOutputClassifier(AdaBoostClassifier(base_estimator=DecisionTreeClassifier())))
        ('multi_output_clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Normally, more parameters could be added for a full gridsearch. To save
    # time and pickle file size, I only gridsearch a few of the better
    # performing parameters 
    parameters = {
        'count_vec__min_df': [2],
        'count_vec__max_df': [0.9],
        'tfidf__use_idf': [True],
        #'tfidf__smooth_idf': [True, False],
        #'tfidf__sublinear_tf': [True, False],
        #'tfidf__norm': ['l1', 'l2'],
        'multi_output_clf__estimator__n_estimators': [50],
        'multi_output_clf__estimator__max_depth': [100],
        'multi_output_clf__estimator__bootstrap': [False]
        
        # For the Adaboost classifier only
        #'multi_output_clf__estimator__base_estimator__n_estimators': [10],
        #'multi_output_clf__estimator__base_estimator__max_depth': [3, 10, 50, 100],
        #'multi_output_clf__estimator__base_estimator__splitter': ['best', 'random'],
        #'multi_output_clf__estimator__base_estimator__max_features': ['auto'],
    }
    
    # Normally, I would train this with a higher cv number, but since this is 
    # mainly for demonstration, and I only have one set of parameters in the 
    # pipeline, I set cv=2 for faster training
    model = GridSearchCV(estimator=pipeline, param_grid=parameters,
                         verbose=3,
                         cv=2
                         )
    
    return model


def evaluate_model(model, X_test, Y_test, category_names,
                  print_=True, return_=False):
    '''
    Evaluate the model performance

    Parameters
    ----------
    model : sklearn GridSearchCV object
        The model to evaluate.
    X_test : pandas dataframe or numpy array
        The design matrix of the test dataset to evaluate.
    Y_test : pandas dataframe or numpy array
        The response values of the test set to evaluate.
    category_names : list
        A list of the category names.
    print_ : bool, optional
        Print the results to the console. The default is True.
    return_ : bool, optional
        Return the results as a dictionary. The default is False.

    Returns
    -------
    reports : dict
        A dictionary of model results.

    '''
    y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)
    y_prob_list = model.predict_proba(X_test)
    y_prob = {cat: 1-y_prob_list[i][:, 0] for i, cat in enumerate(category_names)}
    reports = {}
    for cat in category_names:
        reports[cat] = {}
        reports[cat]['Full Report'] = classification_report(Y_test[cat].values, y_pred[cat].values)
        try:
            reports[cat]['AUC'] = roc_auc_score(Y_test[cat], y_prob[cat])
        except ValueError:
            reports[cat]['AUC'] = 'One class, AUC not defined'
        if print_:
            print('Full Report: \n', reports[cat]['Full Report'])
            print('\tAUC: \n\t\t', reports[cat]['AUC'])
            print('\tAccuracy: \n\t\t', accuracy_score(Y_test[cat].values, y_pred[cat].values), '\n')
    if print_:
        print('\nOverall Accuracy: ', np.mean(Y_test.values == y_pred.values))
    if return_:
        return reports
    else:
        return None


def save_model(model, model_filepath):
    '''
    Save the model to a pickle file

    Parameters
    ----------
    model : sklearn GridSearchCV object
        The model to save.
    model_filepath : str
        The directory and filename (path) to save the model to.

    Returns
    -------
    None.

    '''
    with open(model_filepath, 'wb') as f:
        joblib.dump(model, f, compress=4)


def main():
    '''
    The main() function loads the disaster response data from the sqlite
    database, trains the model, evaluates the model, and saves the model

    Returns
    -------
    None.

    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                            Y,
                                                            test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()