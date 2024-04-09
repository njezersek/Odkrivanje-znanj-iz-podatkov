import wikipediaapi
from unidecode import unidecode

languages = ["en", "de", "sv", "nl", "da",
             "fr", "es", "it", "pt", "ca",
             "ru", "pl", "uk", "sr", "sl",
             "fi", "hu", "et", "el", "ja"]

topics = ["finland", "russia", "japan", "italy", "united states", "china", "wikipedia", "africa", "eruope"]

all_texts = {x: [] for x in languages}

def get_from_topic(topic):
    wiki = wikipediaapi.Wikipedia("en")

    page = wiki.page(topic)
    text = page.text
    n = 0

    if len(unidecode(text)) > 10000 and len(all_texts["en"]) < 1:
        all_texts["en"].append((topic, text))
        print(f"en: {topic} has {len(text)} characters")
        n += 1

    for lang in page.langlinks:        
        if lang in languages:
            if len(all_texts[lang]) == 5:
                continue

            temp_wiki = wikipediaapi.Wikipedia(lang)
            title = page.langlinks[lang].title
            text = temp_wiki.page(title).text
            
            if len(unidecode(text)) > 10000 and len(all_texts[lang]) < 5:
                all_texts[lang].append((topic, text))
                print(f"{lang}: {title} has {len(text)} characters")
                n += 1

    print(f"Got {n} articles for {topic}")

for topic in topics:
    get_from_topic(topic)

for lang, texts in all_texts.items():
    for title, text in texts:
        with open(f"jeziki/{lang}-{title.replace(' ', '')}.txt", "w", encoding="utf-8") as f:
            f.write(text)