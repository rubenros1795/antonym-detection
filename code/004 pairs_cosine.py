from imports import *

model = fasttext.load_model(MODEL_PATH)
chunks = gb(PAIR_CHUNK_PATH + '/*')

def calc_cos_sim(w1, w2):
    v1 = model.get_word_vector(w1)
    v2 = model.get_word_vector(w2)
    return cosine_similarity([v1],[v2])[0][0]

df = set()
for chunk in tqdm(chunks):
    with open(chunk,'r') as f:
        j = json.load(f)
    for i in j:
        for id_, vls in i.items():
            for w1, w2 in vls:
                y = (w1,w2)
                y = tuple(sorted(y))
                df.add(y)

with open(os.path.join(DATA_PATH,'total-pairs.json'),'w') as f:
    json.dump(list(df),f)

with open(os.path.join(DATA_PATH,'total-pairs-similarity'),'w') as f:
    for w1, w2 in tqdm(df):
        s = calc_cos_sim(w1,w2)
        f.write('\t'.join([w1,w2,str(round(s,4))]) + '\n')

df = pd.read_csv(os.path.join(DATA_PATH,'total-pairs-similarity'),sep='\t')
df.columns = ['w1','w2','s']
df.to_parquet(os.path.join(DATA_PATH,'total-pairs-similarity.parquet'))
tr = pd.read_csv(os.path.join(DATA_PATH,'pairs-vd-mwb-full.tsv'),sep='\t')
sims = [calc_cos_sim(r['w1'],r['w2']) for i,r in tqdm(tr.iterrows())]
tr['cosine'] = sims
tr.to_parquet(os.path.join(DATA_PATH,'pairs-vd-mwb-cosine.parquet'))