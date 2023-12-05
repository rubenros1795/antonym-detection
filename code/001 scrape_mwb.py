from ..imports import *

class MijnWoordenboekScraper:
    def __init__(self, relation):
        self.relation = relation
        self.result = []

    def get_words_page(self, char, n):
        url = f"https://www.mijnwoordenboek.nl/{self.relation}/{char}/{n}.html"
        soup = bs(rq.get(url).content)
        word_urls = [(a['href'], a['href'].split('/')[-1]) for a in soup.find_all("a", href=True)
                     if f'/{self.relation}/' + char.lower() in a['href']]
        return word_urls

    def parse_all(self):
        for char in tqdm(afb):
            for n in range(1, 21):
                word_urls = self.get_words_page(char, n)
                if len(word_urls) > 0:
                    try:
                        for url, word in word_urls:
                            soup = bs(rq.get(url).content)
                            for word2 in [t.text for t in soup.find('ul', class_='icons-ul').find_all('a', href=True)]:
                                self.result.append({"word1": word, "word2": word2, "relation": self.relation})
                    except:
                        continue
                else:
                    continue

        df = pd.DataFrame(self.result)
        df = df.groupby(df.apply(frozenset, axis=1), as_index=False).first()
        return df

def main():
    syn_scraper = MijnWoordenboekScraper("synoniemen")
    ant_scraper = MijnWoordenboekScraper("antoniemen")

    synonyms = syn_scraper.parse_all()
    antonyms = ant_scraper.parse_all()

    pairs = pd.concat([synonyms, antonyms])
    pairs['relation'] = pairs.relation.apply(lambda x: "syn" if x == "synoniemen" else "ant")
    
    pairs.to_csv('pairs-as.tsv', sep='\t', index=False)

    pairs = pd.concat([synonyms.sample(len(antonyms)), antonyms])
    pairs['relation'] = pairs.relation.apply(lambda x: "S" if x == "synoniemen" else "A")
    
    pairs.to_csv(os.path.join(TRAINING_DATA_PATH,'pairs-as-nh.csv'), sep=',', index=False, header=False)

if __name__ == "__main__":
    main()
