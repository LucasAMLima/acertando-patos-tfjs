"""
Exporta o checkpoint treinado (best.pt) para TensorFlow.js com o export.py do
repositório ultralytics/yolov5 e copia o resultado para a extensão.

  python export_tfjs.py --weights <runs>/ducks/weights/best.pt

O "--include tfjs" do YOLOv5 embute o NMS no grafo, então o modelo no navegador
já devolve [boxes, scores, classes, count].
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--conf-thres', default='0.25')
    args = ap.parse_args()

    weights = Path(args.weights).resolve()
    env = dict(os.environ)
    env['TF_USE_LEGACY_KERAS'] = '1'  # o export TF do YOLOv5 usa a API Keras 2 (tf_keras)
    # export.py chama "tensorflowjs_converter" por nome: garante que o do venv está no PATH
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')

    subprocess.run([sys.executable, str(HERE / 'yolov5' / 'export.py'), '--weights', str(weights),
                    '--include', 'tfjs', '--img', '640', '--conf-thres', args.conf_thres],
                   check=True, env=env)

    src = weights.with_name(weights.stem + '_web_model')
    if not (src / 'model.json').exists():
        sys.exit(f'export falhou: {src}/model.json não encontrado')
    dst = HERE.parent / 'extension' / 'models' / 'ducks_web_model'
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    print(f'Modelo TF.js copiado para {dst}')


if __name__ == '__main__':
    main()
