import re


def preprocess(text):
    text = re.sub(r'(Page\s*\d+|=*\s*Slide\s*\d+\s*=*|<LINK>)', '', text)
    text = text.encode("ascii", errors="ignore").decode()
    text = " ".join(re.split(r'[\n\t]+', text)).strip()
    text = re.sub(r'(,\s*){2,}', ',', text)
    return text[:1000000]