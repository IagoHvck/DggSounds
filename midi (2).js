DggSounds – Explorando o Áudio Digital
Protótipo de Sistema Multimídia (Unidade 2)

COMO ABRIR
  Abra o arquivo index.html no Chrome, Edge ou Firefox (funciona offline, direto da pasta).

ESTRUTURA
  index.html          páginas e navegação não-linear (#inicio, #som, #digital, #tamanho, #formatos, #edicao, #midi, #quiz, #sobre)
  css/estilo.css      aparência e animações (CSS3)
  js/app.js           interatividade, Web Audio API, animações em canvas, quiz
  js/midi.js          leitor de arquivo MIDI + sintetizador em JavaScript
  js/quiz.js          banco de 20 perguntas (15 múltipla escolha + 5 V/F); 5 sorteadas por rodada
  js/dados.js         MIDI embutido e tamanhos reais dos arquivos (gerado automaticamente)
  audio/              áudio digitalizado (WAV, FLAC, MP3) e efeitos sonoros
  midi/               trilha_menu.mid (música MIDI)
  img/                imagens matriciais (PNG/JPG)
  svg/                imagens vetoriais (logo, diagramas)
  video/              vídeo-aula em MP4
  tools/              scripts que geraram as mídias (material "não editado" -> "editado")

O QUE O GRUPO AINDA PRECISA COLOCAR
  1. Fotos dos equipamentos: substituir img/foto_microfone.jpg, img/foto_celular.jpg e
     img/foto_computador.jpg por fotos reais (mesmo nome, formato 4:3).
  2. Narrações: gravar no Audacity e salvar como audio/narracao_som.mp3, narracao_digital.mp3,
     narracao_tamanho.mp3, narracao_formatos.mp3, narracao_edicao.mp3, narracao_midi.mp3
     (o texto a narrar é o parágrafo de abertura de cada módulo). Enquanto o arquivo não
     existir, o botão "Ouvir narração" usa a voz sintetizada do navegador.
