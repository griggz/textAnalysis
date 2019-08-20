"""
Provides top key words from text chunk.

text_analysis.py was built to analyze chunks of text. More specifically, reviews scraped from the web. It is passed a business
slug to identify data from the database. FrequencyDistribution holds a mapping from words to counts, and functions such as
tokenize() create lists of words that can be mapped. 

Next steps: Generalize this script further by stripping out database functions.
"""
import argparse
from django.conf import settings
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from scrape.models import Yelp, Results
import spacy
from spacy.lang.en import English

stop_words = set(stopwords.words('english'))
spacy.load('en')
parser = English()


class AnalyzeText:
    def __init__(self, biz_slug):
        self.biz_slug = biz_slug
        self.words_ignore = self.gen_ignore_words()
        self.originData = self.gather_data()

    def clean_data(self):
        """This merges all reviews into a single string and removes all stop
        and non essential words from the ignore_json_file."""
        sentence = self.originData
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(sentence)
        filtered_words = [w for w in tokens if w not in stop_words and w not in self.words_ignore]
        data = " ".join(filtered_words)
        return self.tokenize(data)

    def gather_data(self):
        """This gathers data from the database based on the object's slug"""
        yelp_id = [x for x in Yelp.objects.filter(slug=self.biz_slug).values_list('id', flat=True)].pop()

        results = [x.lower() for x in Results.objects.filter(business=yelp_id).values_list('review', flat=True)]

        return ' '.join(results)

    def gen_ignore_words(self):
        """This compiles words I have tagged as 'non essential'. They are
        currently being stored in a json file I am manually updating"""
        df = pd.read_json(settings.IGNORE_WORDS_JSON)

        return [x for x in df['ignore']]

    @staticmethod
    def tokenize(text):
        """turns a single string of multiple words into list of words"""
        lda_tokens = []
        tokens = parser(text)
        for token in tokens:
            if token.orth_.isspace():
                continue
            elif token.like_url:
                lda_tokens.append('URL')
            elif token.orth_.startswith('@'):
                lda_tokens.append('SCREEN_NAME')
            else:
                lda_tokens.append(token.lower_)
        return lda_tokens


class ProcessCommon(AnalyzeText):
    def __init__(self, biz_slug, *args):
        super().__init__(biz_slug)
        self.words = self.clean_data()

    def analyze_data(self):
        """word frequencies are generated and are compared against word lengths"""
        words = nltk.FreqDist(self.words)
        most_common = [x[0] for x in words.most_common(50)]

        # sort words by the length of word and then by the number of occurrences in the doc
        # most_common_alt = list(w for w in set(self.words) if len(w) >= 5 and words[w] >= 5)
        most_common_alt = list(w for w in set(self.words) if len(w) >= 5 and w in most_common)
        print(len(most_common), most_common)
        print(len(most_common_alt), most_common_alt)

    def run(self):
        self.analyze_data()


# class ProcessSentiment(AnalyzeText): (UNDER DEV)
#     def __init__(self, biz_slug, *args):
#         super().__init__(biz_slug)
#         self.words = self.clean_data()


def main():
    parser = argparse.ArgumentParser(description='business slug')

    parser.add_argument('-s', '--slug', required=True,
                        help='Dont forget the biz slug!'),

    args = vars(parser.parse_args())

    ProcessCommon(args['slug']).run()


if __name__ == '__main__':
    main()
