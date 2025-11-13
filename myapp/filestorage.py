import os, json
from django.conf import settings
DATA_DIR = os.path.join(settings.BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def save_data(filename, data):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_data(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r', encoding='utf8') as f:
        return json.load(f)

if not load_data('courses.json'):
    print("Creating default courses.json")
    save_data('courses.json', {'courses': ['data', 'more data']})