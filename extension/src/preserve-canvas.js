// Roda no "MAIN world" ANTES do jogo carregar (run_at: document_start).
//
// O jogo (Construct 2) desenha num canvas WebGL criado sem
// `preserveDrawingBuffer`. Nesse modo o navegador limpa o buffer depois de
// cada frame e, ao lermos os pixels (tf.browser.fromPixels), vem uma imagem
// preta. Forçamos a opção para que a IA consiga "enxergar" a tela.
(() => {
  const originalGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (type, attrs) {
    if (typeof type === 'string' && type.includes('webgl')) {
      attrs = { ...(attrs || {}), preserveDrawingBuffer: true };
    }
    return originalGetContext.call(this, type, attrs);
  };
})();
