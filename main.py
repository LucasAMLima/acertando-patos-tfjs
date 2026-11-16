"""
Demonstração: abre o Duck Challenge no SITE DO JOGO
(https://www.jogos123.net/jogo/duck-challenge) e deixa a IA (YOLOv5n +
TensorFlow.js) jogar sozinha.

Nada do jogo é baixado nem servido localmente: a página é a do próprio site. O
jogo roda num iframe vindo de cdn.games.jogos123.net, e é nesse iframe que o
main.py injeta os MESMOS scripts da extensão (extension/):
  - src/preserve-canvas.js   antes do jogo criar o WebGL (init script)
  - lib/tf.min.js + src/duck-bot.js   depois que o canvas do jogo existe
O modelo (extension/models/ducks_web_model) é entregue ao iframe interceptando
as requisições para https://cdn.games.jogos123.net/__duck-bot/... (mesma origem
do jogo, então não há problema de CORS).

Por que não carregar a extensão direto: o Chrome/Edge atuais ignoram
--load-extension. Para usar a extensão de verdade, instale-a no seu Chrome
(veja o README) e abra o site normalmente.

Instalação (uma vez):
  pip install -r requirements.txt
  python -m playwright install chromium     # só se você não tiver Chrome nem Edge

Uso:
  python main.py                         # abre o navegador e joga até você fechar a janela
  python main.py --seconds 120           # joga 2 minutos e fecha
  python main.py --headless --seconds 90 --screenshots test/out   # sem janela (teste E2E)
  python main.py --no-autoplay           # você clica em Play; o bot só atira
"""
import argparse
import json
import mimetypes
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXT = ROOT / 'extension'
SITE_URL = 'https://www.jogos123.net/jogo/duck-challenge'
GAME_ORIGIN = 'https://cdn.games.jogos123.net'
GAME_PATH = '/html5/duck-challenge/'
ASSET_PREFIX = '/__duck-bot/'  # caminho "virtual" que o main.py atende com arquivos de extension/

# Roda no mundo principal de todos os frames desde o início do carregamento,
# mas só age dentro do iframe do jogo.
PRESERVE_INIT = """
if (location.href.includes('%s')) {
  window.DUCK_BOT_MODEL_URL = '%s%smodels/ducks_web_model/model.json';
  %s
}
""" % (GAME_PATH, GAME_ORIGIN, ASSET_PREFIX, (EXT / 'src' / 'preserve-canvas.js').read_text(encoding='utf8'))

# Quando o jogo está no menu (início ou depois de um game over) e o bot já
# carregou o modelo, espera 3 s e clica em Play — a demonstração roda em loop.
AUTOPLAY_INIT = """
(() => {
  if (!location.href.includes('%s')) return;
  let menuSince = 0;
  setInterval(() => {
    const rt = window.cr_getC2Runtime && cr_getC2Runtime();
    const layout = rt && rt.running_layout && rt.running_layout.name;
    const botReady = !!document.documentElement.dataset.duckBot;
    if (layout !== 'StartMenu' || !botReady) { menuSince = 0; return; }
    if (!menuSince) { menuSince = performance.now(); return; }
    if (performance.now() - menuSince < 3000) return;
    menuSince = 0;
    const c = document.getElementById('c2canvas'), r = c.getBoundingClientRect();
    const o = { bubbles: true, cancelable: true, view: window, button: 0,
                clientX: r.left + r.width * 0.5, clientY: r.top + r.height * 0.555 };
    for (const t of ['mousemove', 'mousedown', 'mouseup', 'click']) c.dispatchEvent(new MouseEvent(t, o));
  }, 500);
})();
""" % GAME_PATH

# Lê o estado do jogo (variáveis globais do Construct 2) e as estatísticas que o
# duck-bot.js publica em <html data-duck-bot="...">.
READ_STATE_JS = """
() => {
  const rt = window.cr_getC2Runtime && cr_getC2Runtime();
  const vars = {};
  if (rt) for (const v of rt.all_global_vars) vars[v.name] = v.data;
  const bot = document.documentElement.dataset.duckBot;
  return { injected: !!document.documentElement.dataset.duckBotInjected,
           layout: rt && rt.running_layout && rt.running_layout.name,
           score: vars.Score, seconds: vars.Seconds, best: vars.BestScore,
           bot: bot ? JSON.parse(bot) : null };
}
"""


def serve_extension_file(route):
    """Atende /__duck-bot/<arquivo> com o conteúdo de extension/<arquivo>."""
    path = route.request.url.split(ASSET_PREFIX, 1)[1].split('?')[0]
    file = (EXT / path).resolve()
    if EXT not in file.parents or not file.is_file():
        return route.fulfill(status=404, body='not found')
    route.fulfill(status=200, body=file.read_bytes(),
                  content_type=mimetypes.guess_type(file.name)[0] or 'application/octet-stream')


def launch(p, headless):
    """Usa o Chrome instalado; se não houver, o Edge; por último o Chromium do Playwright."""
    args = ['--autoplay-policy=no-user-gesture-required']
    args += ['--enable-unsafe-swiftshader'] if headless else ['--start-maximized']
    errors = []
    for channel in ('chrome', 'msedge', None):
        try:
            return p.chromium.launch(channel=channel, headless=headless, args=args)
        except Exception as e:  # noqa: BLE001 - tenta o próximo navegador
            errors.append(f'{channel or "chromium"}: {str(e).splitlines()[0]}')
    sys.exit('Não consegui abrir nenhum navegador:\n  ' + '\n  '.join(errors)
             + '\nInstale o Chrome ou rode: python -m playwright install chromium')


def inject_bot(frame):
    frame.wait_for_selector('#c2canvas', timeout=60000)
    frame.evaluate('document.documentElement.dataset.duckBotInjected = "1"')
    frame.add_script_tag(content=(EXT / 'lib' / 'tf.min.js').read_text(encoding='utf8'))
    frame.add_script_tag(content=(EXT / 'src' / 'duck-bot.js').read_text(encoding='utf8'))


def run(args):
    try:
        from playwright.sync_api import sync_playwright, Error as PlaywrightError
    except ImportError:
        sys.exit('Playwright não instalado. Rode: pip install -r requirements.txt')

    if not (EXT / 'models' / 'ducks_web_model' / 'model.json').exists():
        sys.exit('Modelo não encontrado em extension/models/ducks_web_model (rode training/export_tfjs.py).')

    shots_dir = Path(args.screenshots) if args.screenshots else None
    if shots_dir:
        shots_dir.mkdir(parents=True, exist_ok=True)

    summary = {'best': 0, 'shots': 0, 'frames': 0, 'detections': {}, 'game_over': False}
    with sync_playwright() as p:
        browser = launch(p, args.headless)
        ctx = browser.new_context(no_viewport=not args.headless,
                                  viewport={'width': 1400, 'height': 900} if args.headless else None)
        ctx.route(f'{GAME_ORIGIN}{ASSET_PREFIX}**', serve_extension_file)
        ctx.add_init_script(PRESERVE_INIT)
        if not args.no_autoplay:
            ctx.add_init_script(AUTOPLAY_INIT)
        page = ctx.new_page()
        print(f'Abrindo {SITE_URL} ({browser.browser_type.name} {browser.version})')
        page.goto(SITE_URL, wait_until='domcontentloaded', timeout=60000)
        try:
            page.locator('#html5-game-frame').scroll_into_view_if_needed(timeout=10000)
        except PlaywrightError:
            pass
        print('A IA joga sozinha (tecla B dentro do jogo liga/desliga o bot). '
              + (f'Rodando por {args.seconds} s.' if args.seconds else 'Feche a janela ou Ctrl+C para sair.'))

        start = time.time()
        last_seconds = None
        try:
            while not args.seconds or time.time() - start < args.seconds:
                page.wait_for_timeout(2000)
                t = int(time.time() - start)
                frame = next((f for f in page.frames if GAME_ORIGIN + GAME_PATH in f.url), None)
                if not frame:
                    print(f't={t:>4}s aguardando o iframe do jogo...')
                    continue
                try:
                    s = frame.evaluate(READ_STATE_JS)
                    if not s['injected']:
                        print(f't={t:>4}s injetando TensorFlow.js + duck-bot.js no iframe do jogo...')
                        inject_bot(frame)
                        continue
                except PlaywrightError as e:
                    if 'closed' in str(e).lower():
                        raise
                    continue  # o iframe pode estar recarregando
                bot = s['bot'] or {}
                summary['best'] = max(summary['best'], s['score'] or 0, s['best'] or 0)
                summary['shots'] = max(summary['shots'], bot.get('shots', 0))
                summary['frames'] = max(summary['frames'], bot.get('frames', 0))
                summary['detections'] = bot.get('detections', summary['detections'])
                if last_seconds and s['seconds'] == 0:
                    summary['game_over'] = True
                last_seconds = s['seconds']
                print(f"t={t:>4}s tela={s['layout']} placar={s['score']} tempo={s['seconds']} "
                      f"recorde={s['best']} | bot: {bot.get('status', 'carregando modelo...')} "
                      f"fps={bot.get('fps', 0)} tiros={bot.get('shots', 0)} "
                      f"prateleiras={' '.join(bot.get('lanes', []))}")
                if shots_dir:
                    page.screenshot(path=str(shots_dir / f't{t:04d}.png'))
        except KeyboardInterrupt:
            print('\nEncerrado.')
        except PlaywrightError as e:
            if 'closed' not in str(e).lower():
                raise
            print('Janela fechada.')
        finally:
            try:
                browser.close()
            except PlaywrightError:
                pass

    print('RESUMO:', json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    ap = argparse.ArgumentParser(description='Demo: YOLOv5n (TF.js) jogando Duck Challenge no site do jogo')
    ap.add_argument('--seconds', type=int, default=0, help='tempo de execução (0 = até fechar a janela)')
    ap.add_argument('--headless', action='store_true', help='roda sem janela (WebGL na CPU, bem mais lento)')
    ap.add_argument('--screenshots', help='pasta para salvar um screenshot a cada 2 s')
    ap.add_argument('--no-autoplay', action='store_true', help='não clica em Play automaticamente')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
