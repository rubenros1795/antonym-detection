from imports import *

df = pd.read_csv(os.path.join(DATA_PATH,'pmi-ants.csv'))
df = df[df.fab > 10]

scores = pd.read_csv(os.path.join(DATA_PATH,'classifier-scores.csv'),index_col=None,usecols=['w1','w2','ant','s'])
scores = {w1+'-'+w2:s for w1,w2,s in zip(scores.w1,scores.w2,scores.ant)}
scores_keys = set(scores.keys())
scores_df = {ap:scores.get(ap) if ap in scores_keys else scores.get(ap.split('-')[1] + '-' + ap.split('-')[0])  for ap in df.ap.unique()}
df['score'] = df.ap.apply(lambda x: scores_df.get(x))


def decay_function(x, c):
    return np.exp(c * x)

constants = {}

for ap,d in df.groupby('ap'):
    X,Y = d.sent_len.values, d.pmi.values 
    popt, _ = curve_fit(decay_function, X, Y)
    estimated_constant = popt[0]
    constants.update({ap:estimated_constant})


df['npmi'] = df.apply(lambda row: row['pmi'] * decay_function(row['sent_len'], constants[row['ap']]), axis=1)
df.to_csv(os.path.join(DATA_PATH,'diachronic-adjusted-pmi-with-scores.csv'),index=False)