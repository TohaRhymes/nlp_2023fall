from typing import List

import demoji
import re
import spacy

NLP = spacy.load("ru_core_news_sm")
CHECK_POS = {'PUNCT', 'ADP', 'AUX', 'CCONJ', 'SCONJ'}
PATTERN_EMAIL = r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+'


def normalize_text(text: str) -> str:
    checked_text = demoji.replace(text)
    checked_text = re.sub(PATTERN_EMAIL, '<EMAIL>', checked_text)
    return checked_text


def normalize_data(text_data: List[str]) -> List[str]:
    return list(map(normalize_text, text_data))


def tokenize_clean_stem(text: str) -> List[str]:
    text_NLPed = NLP(text)
    return [token.lemma_ for token in text_NLPed
            if token.pos_ not in CHECK_POS and not token.is_stop]


def tokenize_clean_stem_data(text_data: List[str]) -> List[List[str]]:
    return list(map(tokenize_clean_stem, text_data))


def pipeline_preprocessing(text_data: List[str]) -> List[List[str]]:
    data = normalize_data(text_data)
    return tokenize_clean_stem_data(data)
