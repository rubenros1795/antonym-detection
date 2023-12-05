from imports import *

class AntonymAnalyzer:
    def __init__(self, antonyms_df, nrc_clean_df):
        self.antonyms_df = antonyms_df
        self.nrc_clean_df = nrc_clean_df

    def _preprocess_year_slice(self, date):
        tdf = self.nrc_clean_df[self.nrc_clean_df.date == date].reset_index(drop=True)
        tdf['clean'] = tdf.clean.str.split('[[S]]', regex=False)
        tdf = tdf.explode(column='clean')
        tdf['sent_len'] = tdf.clean.apply(lambda x: len(x.split(' ')))
        return tdf

    def _preprocess_wv(self, tdf, sentence_length):
        tdf = tdf[tdf.sent_len > sentence_length]
        vect = CountVectorizer()
        wv = vect.fit_transform(raw_documents=tdf.clean.astype(str))
        return tdf, vect, wv

    def _create_term_sums(self, matrix):
        term_sums = np.sum(matrix, axis=0)
        term_sum_dict = {term_index: term_count for term_index, term_count in enumerate(term_sums.A1)}
        return term_sum_dict

    def _count_pair(self, matrix, term_combination):
        binary_matrix = (matrix[:, term_combination] > 0).toarray()
        doc_count = np.sum(np.prod(binary_matrix, axis=1))
        return doc_count

    def analyze_antonyms(self):
        results = []

        for year in tqdm(self.nrc_clean_df.date.unique()):
            try:
                tdf = self._preprocess_year_slice(year)
                sent_props = {sent_len: val / len(tdf) for sent_len, val in tdf.sent_len.value_counts().to_dict().items()}

                for sentence_length in range(3, 18):
                    tdfs, vect, wv = self._preprocess_wv(tdf, sentence_length)
                    tsdict = self._create_term_sums(wv)
                    word_indices = {w: c for c, w in enumerate(vect.get_feature_names_out())}
                    period_sum = sum(tsdict.values())

                    for w1, w2 in self.antonyms_df[['w1', 'w2']].itertuples(index=False):
                        i1, i2 = word_indices.get(w1), word_indices.get(w2)
                        f1, f2 = tsdict.get(i1), tsdict.get(i2)

                        if any(i is None for i in [i1, i2, f1, f2]):
                            continue

                        pa, pb = f1 / period_sum, f2 / period_sum
                        pab = self._count_pair(wv, [i1, i2])

                        if pab == 0 or pab is None:
                            continue

                        pab_ = pab / period_sum
                        results.append({
                            "date": year,
                            "sent_len": sentence_length,
                            "sent_len_prop": sent_props[sentence_length],
                            "date_total": period_sum,
                            "ap": '-'.join([w1, w2]),
                            "fa": f1,
                            "fb": f2,
                            "fab": pab,
                            "pa": pa,
                            "pb": pb,
                            "pab": pab_,
                            "pmi": np.log2((pab_) / (pa * pb))
                        })
            except Exception as e:
                print(f"Error processing year {year}: {e}")

        # Convert results to DataFrame and return
        return pd.DataFrame(results)

## Run
# pairs_df is a tabular file containing pairs of words relating to classified antonyms, along with the article id
# nrc_clean is a tabular file with two columns: url and clean, respectively the url of an article (from which a data is extracted) and the cleaned text
# the result is a

antonyms_df = pd.read_csv(os.path.join(DATA_PATH,"pairs_df.csv"))
nrc_clean_df = pd.read_parquet(os.path.join(DATA_PATH,"nrc_clean.tsv.gzip"))

analyzer = AntonymAnalyzer(antonyms_df, nrc_clean_df)
result_df = analyzer.analyze_antonyms()
result_df.to_csv(os.path.join(DATA_PATH,'pmi-ants.csv'), index=False)
