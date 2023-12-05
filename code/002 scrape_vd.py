from imports import *

class VandaleScraper:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.driver = self.initialize_driver()

    def initialize_driver(self):
        options = Options()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        return driver

    def login(self):
        self.driver.get("https://zoeken.vandale.nl/login")
        time.sleep(3)

        partial_href = '/auth/realms/VanDale/broker/surfconext_idp'
        link = self.driver.find_element("xpath", f'//a[contains(@href, "{partial_href}")]')
        link.click()
        time.sleep(2)

        search_field = self.driver.find_element('id', 'wayf_search')
        search_text = 'Leiden'
        search_field.send_keys(search_text)
        time.sleep(2)

        keyword = 'leiden'
        xpath = f'//li[contains(@data-keywords, "{keyword}")]'
        element = self.driver.find_element("xpath", xpath)
        time.sleep(2)
        element.click()
        time.sleep(2)

        username_field = self.driver.find_element("name", 'Ecom_User_ID')
        username_field.send_keys(self.username)

        password_field = self.driver.find_element("name", 'Ecom_Password')
        password_field.send_keys(self.password)

        login_button = self.driver.find_element("id", 'loginbtn')
        login_button.click()
        time.sleep(3)

        self.driver.get('https://zoeken.vandale.nl/?dictionaryId=gsy')

    def scrape_antonyms(self, words, output_path):
        for w in tqdm(words):
            try:
                input_element = self.driver.find_element('css selector', '.search__input')
                input_element.clear()
                input_element.send_keys(w)
                input_element.send_keys(Keys.ENTER)
                time.sleep(1.8)
                html = self.driver.find_element('id', 'article').get_attribute('innerHTML')
                with open(os.path.join(output_path, w), 'w') as f:
                    f.write(html)
            except Exception as e:
                continue

    def close_driver(self):
        self.driver.close()

def extract_words(word):
    with open(VD_ANTONYM_HTML + word, 'r') as f:
        s = bs(f.read())
    result = []
    lemma = s.find('span', class_='gag').text

    for rel, vds in [("antonym", ""), ("synonym", "")]:
        words = [a.findParent() for a in s.find_all('a') if any(e in a.find('span').attrs['class'] for e in
                                                              ['gb5', 'gaj'])]
        words1 = [w.find_all('span', class_='gaj') for w in words if vds in w.get_text()]
        words1 += [w.find_all('span', class_='gb5') for w in words if vds in w.get_text()]

        words = [item.get_text().replace('\xad', '') for sublist in words1 for item in sublist]
        words = set([w for w in words if w.isalpha()])

        if len(words) > 0:
            for w in words:
                result.append({'w1': word, 'w1-lemma': lemma, 'w2': w, 'rel': rel})
    return result


def main():

    username = 'your_username'  # Replace with your Vandale username
    password = 'your_password'  # Replace with your Vandale password
    scraper = VandaleScraper(username, password)
    
    path_dts = 'sa_datasets'
    missing = pd.concat([pd.read_csv(f, sep='\t', header=None) for f in gb(path_dts + '/*') if 'add' not in f])[
        [0, 1]]
    missing = list(set(missing[[0, 1]].values.flatten()))
    missing = [w for w in missing if ' ' not in w]

    parsed = set([f.split('.')[0] for f in os.listdir(VD_ANTONYM_HTML)])
    missing = [w for w in missing if w not in parsed]
    print(len(missing))

    scraper.login()
    scraper.scrape_antonyms(missing, VD_ANTONYM_HTML)
    scraper.close_driver()

    fs = gb(VD_ANTONYM_HTML + '/*')

    r = []

    for w in tqdm([f.split('/')[-1] for f in fs]):
        r += extract_words(w)

    rd = pd.DataFrame(r)

    rd['source'] = 'vandale'
    mwb = pd.read_csv('pairs-as.tsv', sep='\t')
    mwb.columns = ['w1', 'w2', 'rel']
    mwb['w1-lemma'] = None
    mwb['rel'] = mwb.rel.apply(lambda x: "synonym" if x == 'syn' else "antonym")
    mwb = mwb[['w1', 'w1-lemma', 'w2', 'rel']]
    mwb['source'] = 'mijnwoordenboek'
    df = pd.concat([rd, mwb])
    df = df.groupby(df[['w1', 'w2']].apply(frozenset, axis=1), as_index=False).first()
    df.to_csv(os.path.join(TRAINING_DATA_PATH,'pairs-vd-mwb-full.tsv'), sep='\t', index=False)

if __name__ == "__main__":
    main()
