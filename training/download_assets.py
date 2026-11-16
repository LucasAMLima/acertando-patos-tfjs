"""Baixa o data.js e as sprite sheets do Duck Challenge para ./assets."""
import re
import urllib.request
from pathlib import Path

BASE = 'https://cdn.games.jogos123.net/html5/duck-challenge/'
OUT = Path(__file__).parent / 'assets'


def get(path: str) -> bytes:
    req = urllib.request.Request(BASE + path, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def main():
    (OUT / 'images').mkdir(parents=True, exist_ok=True)
    data = get('data.js')
    (OUT / 'data.js').write_bytes(data)
    images = sorted(set(re.findall(r'"(images/[\w\-]+\.png)"', data.decode('utf8'))))
    for p in images:
        (OUT / p).write_bytes(get(p))
    print(f'{len(images)} imagens salvas em {OUT}')


if __name__ == '__main__':
    main()
