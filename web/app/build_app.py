"""Embed landmark/model_bundle.json into template.html -> index.html.
Run after train_landmark_mlp.py:  python build_app.py"""
from pathlib import Path
H = Path(__file__).resolve().parent
tpl = (H / 'template.html').read_text(encoding='utf-8')
bundle = (H / 'landmark' / 'model_bundle.json').read_text(encoding='utf-8')
head = ('<!doctype html>\n<html lang="vi">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n</head>\n<body>\n')
(H / 'index.html').write_text(head + tpl.replace('__BUNDLE__', bundle) + '\n</body>\n</html>\n', encoding='utf-8')
print('index.html written')
