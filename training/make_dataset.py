"""
Gera um dataset SINTÉTICO (formato YOLO) para o jogo Duck Challenge.

Ideia: o YOLOv5 pré-treinado no COCO não reconhece os patinhos de borracha
do jogo (eles são desenhos, não pássaros reais). Em vez de rotular imagens à
mão, usamos os próprios sprites do jogo: o arquivo data.js do Construct 2
descreve exatamente onde cada frame de animação está nas sprite sheets. Assim
colamos patos/ovos em posições aleatórias sobre o fundo e já sabemos a caixa
(bounding box) de cada um -> rótulos perfeitos e de graça.

Classes:
  0 = duck      (pato AMARELO com alvo: deve ser atingido)
  1 = red_duck  (pato VERMELHO: NÃO atirar)
  2 = egg       (ovos/pintinhos: NÃO atirar)

Uso:
  python download_assets.py                      # baixa data.js e imagens
  python make_dataset.py --assets assets --out datasets/ducks
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

CLASSES = ['duck', 'red_duck', 'egg']
OUT_W, OUT_H = 960, 540  # metade do canvas original (1920x1080)
SCALE = OUT_W / 1920


def walk(node, fn):
    if isinstance(node, list):
        fn(node)
        for child in node:
            walk(child, fn)


def load_sprites(assets: Path):
    """Lê data.js e recorta cada frame das animações de Duck (t25) e Eggs (t28)."""
    project = json.loads((assets / 'data.js').read_text(encoding='utf8'))['project']
    sheets = {}
    sprites = {'duck': [], 'red_duck': [], 'egg': [], 'duck_die': []}

    def crop(frame):
        path, _, x, y, w, h = frame[:6]
        if path not in sheets:
            sheets[path] = Image.open(assets / path).convert('RGBA')
        img = sheets[path].crop((x, y, x + w, y + h))
        bbox = img.getchannel('A').point(lambda a: 255 if a > 40 else 0).getbbox()
        return img.crop(bbox) if bbox else None

    def visit(node):
        if not node or node[0] not in ('t25', 't28') or len(node) < 8:
            return
        for anim in node[7]:
            name, frames = anim[0], anim[7]
            for i, f in enumerate(frames):
                img = crop(f)
                if img is None or min(img.size) < 6:
                    continue
                if node[0] == 't28':
                    sprites['egg'].append(img)
                elif name.endswith('_Die'):
                    # Os primeiros frames da morte são quase iguais ao pato vivo:
                    # ficam fora para não confundir o modelo. Os demais (pato
                    # "girando" e ficando fino) entram como distrator sem rótulo.
                    if i >= 4:
                        sprites['duck_die'].append(img)
                else:
                    sprites['red_duck' if is_red(img) else 'duck'].append(img)

    walk(project, visit)
    return sprites


def is_red(img):
    """Distingue o pato vermelho do amarelo pela cor média dos pixels opacos."""
    px = [p for p in img.getdata() if p[3] > 200]
    r = sum(p[0] for p in px) / len(px)
    g = sum(p[1] for p in px) / len(px)
    return g < 0.6 * r


def load_distractors(assets: Path):
    names = ['bullet_hole-sheet0', 'cursor-sheet0', 'cursorhover-sheet0',
             'explosion-sheet0', 'score-sheet0', 'sunlight-sheet0', 'gameover-sheet0',
             'gamelogo-sheet0', 'btnplay-sheet0']
    out = []
    for n in names:
        p = assets / 'images' / f'{n}.png'
        if p.exists():
            out.append(Image.open(p).convert('RGBA'))
    return out


def make_background(assets: Path, rng: random.Random):
    bg = Image.open(assets / 'images' / 'background-sheet0.png').convert('RGB')
    # desloca verticalmente (as prateleiras "rolam" no início da partida)
    dy = rng.randint(-300, 300)
    canvas = Image.new('RGB', bg.size)
    canvas.paste(bg, (0, dy))
    canvas.paste(bg, (0, dy - bg.height if dy > 0 else dy + bg.height))
    top = assets / 'images' / 'curtaintop-sheet0.png'
    if top.exists() and rng.random() < 0.5:
        ct = Image.open(top).convert('RGBA')
        canvas.paste(ct, (0, rng.randint(-540, 0)), ct)
    canvas = canvas.resize((OUT_W, OUT_H), Image.BILINEAR)
    return ImageEnhance.Brightness(canvas).enhance(rng.uniform(0.8, 1.15))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def paste(canvas, sprite, rng, scale_range=(0.85, 1.25)):
    s = SCALE * rng.uniform(*scale_range)
    w, h = max(4, int(sprite.width * s)), max(4, int(sprite.height * s))
    img = sprite.resize((w, h), Image.BILINEAR)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # pés apoiados numa "prateleira" qualquer; pode sair parcialmente da tela
    x = rng.randint(-w // 2, OUT_W - w // 2)
    y = rng.randint(int(h * 0.6), OUT_H + h // 4) - h
    canvas.paste(img, (x, y), img)
    return (x, y, x + w, y + h), img


def generate(i, sprites, distractors, assets, out_img, out_lbl, seed):
    rng = random.Random(seed)
    canvas = make_background(assets, rng)
    labels, boxes = [], []

    n_objects = rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8]) if i % 15 else 0  # ~7% sem nada
    for _ in range(n_objects):
        cls = rng.choices([0, 1, 2], weights=[0.5, 0.2, 0.3])[0]
        sprite = rng.choice(sprites[CLASSES[cls]])
        for _try in range(10):
            s = SCALE * rng.uniform(0.85, 1.25)
            w, h = int(sprite.width * s), int(sprite.height * s)
            x = rng.randint(-w // 2, OUT_W - w // 2)
            y = rng.randint(int(h * 0.6), OUT_H + h // 4) - h
            box = (x, y, x + w, y + h)
            if all(iou(box, b) < 0.15 for b in boxes):
                break
        img = sprite.resize((w, h), Image.BILINEAR)
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        canvas.paste(img, (x, y), img)
        boxes.append(box)
        # recorta a caixa à área visível e descarta se quase tudo ficou fora
        cx1, cy1 = max(0, x), max(0, y)
        cx2, cy2 = min(OUT_W, x + w), min(OUT_H, y + h)
        if (cx2 - cx1) * (cy2 - cy1) < 0.4 * w * h:
            continue
        labels.append((cls, (cx1 + cx2) / 2 / OUT_W, (cy1 + cy2) / 2 / OUT_H,
                       (cx2 - cx1) / OUT_W, (cy2 - cy1) / OUT_H))

    # distratores sem rótulo: patos morrendo, buracos de bala, mira, explosão...
    for _ in range(rng.randint(0, 3)):
        paste(canvas, rng.choice(sprites['duck_die']), rng)
    for _ in range(rng.randint(0, 6)):
        d = rng.choice(distractors)
        paste(canvas, d, rng, (0.3, 1.0))

    if rng.random() < 0.3:
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    canvas.save(out_img / f'{i:05d}.jpg', quality=rng.randint(70, 95))
    (out_lbl / f'{i:05d}.txt').write_text(
        ''.join(f'{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n' for c, x, y, w, h in labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assets', default='assets')
    ap.add_argument('--out', default='datasets/ducks')
    ap.add_argument('--n-train', type=int, default=1200)
    ap.add_argument('--n-val', type=int, default=150)
    args = ap.parse_args()

    assets, out = Path(args.assets), Path(args.out)
    sprites = load_sprites(assets)
    distractors = load_distractors(assets)
    print({k: len(v) for k, v in sprites.items()}, 'distratores:', len(distractors))

    for split, n, base_seed in (('train', args.n_train, 0), ('val', args.n_val, 10**6)):
        img_dir, lbl_dir = out / 'images' / split, out / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            generate(i, sprites, distractors, assets, img_dir, lbl_dir, base_seed + i)
        print(f'{split}: {n} imagens em {img_dir}')

    (out / 'ducks.yaml').write_text(
        f'path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n'
        f'names:\n' + ''.join(f'  {i}: {n}\n' for i, n in enumerate(CLASSES)))
    print('yaml:', out / 'ducks.yaml')


if __name__ == '__main__':
    main()
