from operator import le
import wikipedia
from itertools import product
from unidecode import unidecode
import glob
import os
import re
import codecs


def creating_files(directory, queries, languages):
    mapa = {}
    min_char = 12000

    for query, lang in product(queries, languages):
        wikipedia.set_lang(lang)
        # print(f"{lang}_{query}.txt")
        query = query.lower()
        try:
            out_text = wikipedia.page(query).content

            num_char = terks(out_text)
            if num_char < min_char:
                # delete file
                print(f"Will not be writing {query} {lang}  TO SHORT, has only {num_char}")
            else:

                print(f"{query} {lang}  LONG enough, has {num_char}.")

                out_text = re.sub('=+', '=', out_text)
                out_text = re.sub('\n+', '\n', out_text)

                selected_text = out_text[:min_char]
                # print(selected_text)
                print(mapa, len(selected_text))

                # nastavljamo jezike v mapa
                if lang in mapa:
                    mapa[lang] += 1
                else:
                    mapa[lang] = 1
                print(mapa)
                if mapa[lang] < 7:
                    file = codecs.open("%s/%s_%s.txt" % (directory, lang, query), "w", "utf-8")
                    file.write(selected_text)
                    file.close()

                else:
                    print(f"{lang} has more then 7 files-> {query} not written.")

        except:
            print(f"For {lang}_{query} is not possible to create a file.")


    return mapa


def terks(text):
    text = unidecode(text).lower()
    text = re.sub('[^A-Za-z ]+', '', text)
    text = re.sub(' +', ' ', text)
    return len(text)


if __name__ == '__main__':
    queries = ['World War I', 'World War II', "French Revolution", "USA"] # dopolni

    languages = ["hi", "si", "hu", "lt", "zh-classical",
                 "en", "de", "da", "sv", "is",
                 "fr", "es", "it", "pt",
                 "hr", "sr", "sk", "sl", "cs", "ru", "pl",
                 "nl", "ja", "vi", "zh", "ar", "uk", "fa", "ca", "id", "hu", "ko"]

    selected_lang = ["hi", "si", "hu", "zh", "ar",
                     "en", "de", "da", "sv", "nl",
                     "fr", "es", "it", "ca", "pt",
                     "uk", "sr", "sl", "cs", "ru", "pl"]

    # creating_files(queries, languages)
    # creating_files(queries, extra_lang)
    gotten = creating_files('jeziki', queries, selected_lang)
    print((len(gotten)))

    # from_lang = {'en': 17, 'de': 15, 'fr': 10, 'sr': 8, 'ru': 16, 'hi': 5,
    #              'pl': 8, 'it': 13, 'sv': 13, 'is': 1, 'es': 11, 'cs': 8,
    #              'si': 8, 'hu': 7, 'da': 6, 'pt': 9, 'hr': 4, 'sl': 5, 'sk': 2, 'lt': 2}
    # from_extra = {'zh': 16, 'fa': 11, 'ko': 3, 'vi': 13, 'uk': 7, 'ca': 11, 'nl': 9, 'id': 14, 'ar': 13, 'pt': 9,
    #               'hu': 7, 'ja': 6}
    #
    # en = ["hi", "si", "hu", "zh", "ar", "en", "de", "da", "sv", "nl", "fr", "es", "it", "ca", "pt", "uk", "sr", "sl",
    #       "cs", "ru", "pl"]
    #
    # final = {'zh': 16, 'de': 15, 'fr': 9, 'sr': 8, 'ru': 15, 'hi': 5, 'pl': 8, 'it': 12, 'ca': 11, 'sv': 13, 'nl': 9,
    #          'es': 11, 'cs': 8, 'si': 8, 'hu': 6, 'da': 6, 'sl': 5}

