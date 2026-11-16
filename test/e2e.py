"""
Teste ponta a ponta: abre o jogo no site (https://www.jogos123.net/jogo/duck-challenge),
deixa a IA jogar por alguns segundos e confere que ela pontuou sem dar game over.

Uso:
  python test/e2e.py                 # 90 s, screenshots em test/out/
  python test/e2e.py 180 pasta_saida

Abre uma janela do navegador: o site recusa navegadores headless (HTTP 403).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 90
out = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parent / 'out')

summary = main.run(SimpleNamespace(seconds=seconds, headless=False, screenshots=out, no_autoplay=False))

checks = {
    'o modelo detectou patos amarelos': summary['detections'].get('duck', 0) > 0,
    'o bot atirou': summary['shots'] > 0,
    'o placar subiu': summary['best'] > 0,
    'não houve game over': not summary['game_over'],
}
for name, ok in checks.items():
    print(('OK   ' if ok else 'FALHOU ') + name)
sys.exit(0 if all(checks.values()) else 1)
