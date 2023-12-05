from imports import *

# Read data
df = pd.read_parquet('nrc_clean.tsv.gzip')

# Create identifier column
df['sentence_id'] = df.groupby(['url']).cumcount() + 1
df['identifier'] = df[['sentence_id', 'url']].astype(str).agg('_'.join, axis=1)
df = df.drop(columns=['sentence_id', 'url'])
df['year'] = df.identifier.str.split('/').str[4]

# Preprocessing function for CountVectorizer (to remove words with numbers)
def preprocess_text(text):
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    text = re.sub(r'\b\w{1,2}\b', '', text)
    return text

# Initialize vectorizer
vect = CountVectorizer(min_df=100, max_df=.95, preprocessor=preprocess_text, token_pattern=r'\b\w+\b')

# Transform text data
X = vect.fit_transform(df.clean.astype(str))

# Save Vocab
vocab_mapping_ = {c: i for c, i in enumerate(vect.get_feature_names_out())}
vocab_mapping = pd.DataFrame(vocab_mapping_.items(), columns=['id_', 'word'])
vocab_mapping.to_parquet('dictionary.csv.gzip', compression='gzip')

# Generate combinations per article
identifiers = df.identifier.tolist()
word_combs = {id_: set(combinations(X[i].indices, 2)) for i, id_ in enumerate(identifiers)}

word_combs = [{id_: [(vocab_mapping_[i], vocab_mapping_[j]) for i, j in cb]} for id_, cb in word_combs.items()]

# Chunk the data
def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

word_combs = list(divide_chunks(word_combs, 10))

# Save chunks as JSON files
for c, chunk in enumerate(word_combs):
    with open(f"chunktest/pairs-chunk-{c}.json", 'w') as f:
        json.dump(chunk, f)
