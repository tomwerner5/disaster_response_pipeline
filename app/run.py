import json
import plotly
import pandas as pd
import re
import warnings
import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
try:
    from sklearn.externals import joblib
except (ImportError, ModuleNotFoundError):
    import joblib
from sqlalchemy import create_engine
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


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


class KmeansTransform(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=36, init_size=1000,
                 batch_size=3000, random_state=42):
        self.n_clusters = n_clusters
        self.init_size = init_size
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_trans = X.copy()
        self.X_trans = X_trans
        self.dist = 1 - cosine_similarity(X_trans)
        self.km = MiniBatchKMeans(n_clusters=self.n_clusters,
                                  init_size=self.init_size,
                                  batch_size=self.batch_size,
                                  random_state=self.random_state)
        self.km.fit(X_trans)
        self.labels = self.km.labels_
        self.ordered_centroids = self.km.cluster_centers_.argsort()[:, ::-1]
        return self.labels


def build_kmeans():    
    pipeline = Pipeline([
        ('count_vec', CountVectorizer(tokenizer=tokenize,
                                      min_df=2,
                                      max_df=0.9)),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('kmeans', KmeansTransform())
    ])
    
    return pipeline


def build_tsne(X, pca_components=36, tsne_components=2,
               print_=True, load_previous=True, save_tsne=True,
               file_loc="tsne.pkl"):
    if load_previous:
        try:
            if print_:
                print("Attempting TSNE Load...")
            tsne = joblib.load("tsne.pkl")
            if print_:
                print("Successful TSNE Load...")
            return tsne
        except (FileNotFoundError, ModuleNotFoundError, AttributeError):
            warnings.warn("""Tried to load previous tsne, but something failed.
                          Either the file wasn't there, or there was a problem
                          with the pickle. Recreating data. Should take
                          about 5 mins.""")
    if print_:
        print("Creating PCA...")
    pca = PCA(n_components=pca_components).fit_transform(X)
    if print_:
        print("Creating TSNE...")
    tsne = TSNE(n_components=tsne_components).fit_transform(pca)
    if save_tsne:
        if print_:
            print("Saving TSNE...")
        with open('tsne.pkl', 'wb') as f:
            joblib.dump(tsne, f, compress=4)
    if print_:
        print("Finished TSNE...")
    return tsne


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

## extract data needed for visuals
# List of features present
feature_list = ['id', 'message', 'original', 'genre']

# Full X and Y
X = df[feature_list]
Y = df.drop(feature_list, axis=1)

category_names = list(Y.columns)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Visual 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Visual 2
    category_counts = Y.T.sum(axis=1)       
    
    # Visual 3
    print("Building Clusters...")
    pipe = build_kmeans()
    
    pipe = pipe.fit(X['message'])
    terms = np.array(pipe.named_steps['count_vec'].get_feature_names())
    clusters = pipe.transform(X['message'])
    
    ordered_centroids = pipe.named_steps['kmeans'].ordered_centroids
    n_clusters = pipe.named_steps['kmeans'].n_clusters
    X_trans = pipe.named_steps['kmeans'].X_trans.todense()
    
    nwords = 10 # number of words per cluster
    
    tsne = build_tsne(X_trans, pca_components=36, tsne_components=2,
                      print_=True, load_previous=True, save_tsne=True)
    
    pop_docs = pd.DataFrame(index=range(1, n_clusters+1),
                            columns=['Top Word ' + str(i) for i in range(1, nwords+1)])
    
    
    for i in range(1, n_clusters+1):    
        pop_docs.loc[i, :] = terms[ordered_centroids[i-1, :nwords]]
    
    pop_docs.index.name = "Cluster"        
    json_docs = pop_docs.to_html(table_id='topwords',
                                 classes="table table-dark")
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=[cat.replace('_', ' ').title() for cat in category_names],
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -45,
                    'automargin': True
                }
            }
        },
        {
            'data': [
                Scatter(
                    x=tsne[:, 0],
                    y=tsne[:, 1],
                    mode='markers',
                    marker_color=clusters,
                    #color=clusters,
                    text=clusters
                )
            ],

            'layout': {
                'title': 'T-SNE Cluster of Documents',
                'yaxis': {
                    'title': "Component 2"
                },
                'xaxis': {
                    'title': "Component 1",

                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON,
                           json_dt=json_docs)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    y_prob = model.predict_proba([query])
    
    category_names = list(Y.columns)
    
    classification_prob = [1-y_prob[i][0, 0] for i in range(len(y_prob))]
    classification_results = dict(zip(category_names, classification_labels))

    # create visuals
    go_graphs = [
        {
            'data': [
                Bar(
                    x=[cat.replace('_', ' ').title() for cat in category_names],
                    y=classification_prob
                )
            ],

            'layout': {
                'title': 'Predicted Probabilities of Categories',
                'yaxis': {
                    'title': "Probability"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -45,
                    'automargin': True
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    go_ids = ["graph-{}".format(i) for i, _ in enumerate(go_graphs)]
    go_graphJSON = json.dumps(go_graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        ids=go_ids,
        graphJSON=go_graphJSON
    )


def main():
    # I was having issues with the default server. Switch the below lines if 
    # needed
    #
    #app.run(host='0.0.0.0', port=3001, debug=True)
    app.run(host='127.0.0.1', port=3001, debug=True)

if __name__ == '__main__':
    main()