#!/usr/bin/env python3

import re
from typing import Set


LOREM_IPSUM_WORDS = {
    'lorem', 'ipsum', 'dolor', 'amet', 'consectetur', 'adipiscing', 'elit',
    'aliquet', 'augue', 'blandit', 'ullamcorper', 'vulputate', 'vivamus',
    'vestibulum', 'suscipit', 'sodales', 'sagittis', 'quisque', 'pulvinar',
    'praesent', 'posuere', 'placerat', 'pellentesque', 'ornare', 'nulla',
    'nisl', 'nisi', 'nibh', 'neque', 'nec', 'nascetur', 'mollis', 'molestie',
    'metus', 'mauris', 'mattis', 'massa', 'maecenas', 'magna', 'luctus',
    'lobortis', 'ligula', 'libero', 'lectus', 'laoreet', 'lacus', 'lacinia',
    'justo', 'interdum', 'integer', 'imperdiet', 'iaculis', 'hendrerit',
    'habitant', 'gravida', 'fusce', 'fringilla', 'faucibus', 'facilisis',
    'euismod', 'etiam', 'erat', 'egestas', 'efficitur', 'duis', 'donec',
    'dictum', 'dignissim', 'diam', 'cursus', 'curabitur', 'cum', 'congue',
    'condimentum', 'commodo', 'bibendum', 'aenean', 'accumsan', 'tempus',
    'tempor', 'tellus', 'semper', 'senectus', 'scelerisque', 'rutrum',
    'risus', 'rhoncus', 'quis', 'quam', 'porttitor', 'porta', 'pharetra',
    'phasellus', 'feugiat', 'fermentum', 'felis', 'fames', 'elementum',
    'eleifend', 'dapibus', 'convallis', 'consequat', 'class', 'aptent',
    'taciti', 'sociosqu', 'litora', 'torquent', 'conubia', 'nostra',
    'inceptos', 'himenaeos', 'proin', 'pretium', 'primis', 'orci', 'nunc',
    'nam', 'morbi', 'malesuada', 'leo', 'ipsum', 'in', 'id', 'hac',
    'habitasse', 'platea', 'dictumst', 'enim', 'cras', 'arcu', 'ante',
    'aliquam', 'ac', 'a', 'ut', 'urna', 'ultrices', 'ultricies', 'turpis',
    'tristique', 'tortor', 'tincidunt', 'volutpat', 'venenatis', 'vehicula',
    'varius', 'sapien', 'sit', 'vel', 'velit', 'vitae', 'viverra'
}


def is_valid_candidate(token: str, stopwords: Set[str]) -> bool:
    token_lower = token.lower()

    if not token.isalpha():
        return False

    if len(token) < 3:
        return False

    if len(token) > 20:
        return False

    if re.match(r'^(.)\1{2,}', token_lower):
        return False

    if token_lower in stopwords:
        return False

    if token_lower in LOREM_IPSUM_WORDS:
        return False

    if re.match(r'^(aa|ee|oo|uu)[a-z]*$', token_lower):
        if len(set(token_lower)) <= len(token) / 2:
            return False

    if re.match(r'^[aeiou]*(gh|rr|hh|nn|ss)+[aeiou]*(gh|rr|hh|nn|ss)*[aeiou]*$', token_lower):
        return False

    if re.match(r'^u+[rghn]+h*$', token_lower):
        return False

    if re.match(r'^a+[rghn]+h*$', token_lower):
        return False

    if re.match(r'^w+o+a+h*$', token_lower):
        return False

    if re.search(r'(.)\1{2,}', token):
        return False

    if re.match(r'^([a-z]{1,3})\1{2,}$', token_lower):
        return False

    if re.match(r'^([a-z])([a-z])\1\2(\1\2?)+$', token_lower):
        return False

    if re.match(r'^[ha]+$', token_lower) and len(token) > 4:
        return False

    if re.match(r'^a{0,2}h{1,2}a{0,2}h+a*h*$', token_lower):
        return False

    if re.match(r'^b+a+h+a+[ha]*$', token_lower):
        return False

    if re.match(r'^(a+r+g+h*|u+g+h+|e+w+|a+w+)$', token_lower):
        return False

    if re.match(r'^y+e+[ahs]+$', token_lower):
        return False

    if re.match(r'^y+e+s+$', token_lower) and len(token) > 3:
        return False

    if re.match(r'^o+h+[ahs]*$', token_lower) and len(token) > 2:
        return False

    if re.match(r'^m+m+[mh]*$', token_lower):
        return False

    if re.match(r'^h+m+[mh]*$', token_lower):
        return False

    if re.match(r'^u+h+[mh]*$', token_lower):
        return False

    if len(set(token_lower)) == 1:
        return False

    if len(token) >= 6 and len(set(token_lower)) <= 2:
        return False

    if len(token) >= 5:
        vowels = sum(1 for c in token_lower if c in 'aeiou')
        if vowels == 0:
            return False

    if re.match(r'^(aa|ee|ii|oo|uu)', token_lower):
        return False

    return True


def load_stopwords():
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words('english'))
    except LookupError:
        print("WARNING: NLTK stopwords not found. Run: python -m nltk.downloader stopwords")
        return set()
