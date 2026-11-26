"""Gera as mídias de áudio, MIDI e imagens matriciais do DggSounds.
Uso: python3 gen_audio.py <pasta_saida>
"""
import sys, os, json, struct, base64, subprocess
import numpy as np
from scipy import signal
from scipy.io import wavfile
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1]
for d in ("audio", "midi", "img", "js"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
rng = np.random.default_rng(42)
SR = 44100

def note_hz(n):
    return 440.0 * 2 ** ((n - 69) / 12)

# ---------- 1. Amostra "violão" por síntese Karplus-Strong ----------
def pluck(freq, dur, sr=SR, decay=0.996):
    N = int(sr / freq)
    buf = rng.uniform(-1, 1, N)
    out = np.zeros(int(sr * dur))
    idx = 0
    for i in range(len(out)):
        out[i] = buf[idx]
        nxt = (idx + 1) % N
        buf[idx] = decay * 0.5 * (buf[idx] + buf[nxt])
        idx = nxt
    env = np.ones_like(out)
    r = int(0.03 * sr)
    env[-r:] = np.linspace(1, 0, r)
    return out * env

melody = [  # (nota midi, início em tempos, duração em tempos)
    (64, 0, 1), (67, 1, 1), (71, 2, 1), (72, 3, 2), (71, 5, 1), (67, 6, 1), (64, 7, 1),
    (62, 8, 1), (65, 9, 1), (69, 10, 1), (72, 11, 2), (69, 13, 1), (65, 14, 2),
    (60, 16, 1), (64, 17, 1), (67, 18, 1), (72, 19, 1), (76, 20, 4),
]
bass = [(45, 0, 8), (41, 8, 8), (48, 16, 8)]
BEAT = 0.3
LEAD = 0.6  # silêncio inicial (útil para medir o ruído na edição)
total = LEAD + 24 * BEAT + 1.2
clean = np.zeros(int(total * SR))
for n, s, d in melody + bass:
    seg = pluck(note_hz(n), d * BEAT + 0.8) * (0.55 if n < 50 else 0.8)
    i = int((LEAD + s * BEAT) * SR)
    clean[i:i + len(seg)] += seg[: len(clean) - i]
clean /= np.max(np.abs(clean)) * 1.12  # pico ~ -1 dBFS


def save_wav(path, x, sr, bits=16):
    x = np.clip(x, -1, 1)
    if bits == 16:
        wavfile.write(path, sr, (x * 32767).astype(np.int16))
    elif bits == 8:
        wavfile.write(path, sr, ((x * 127) + 128).astype(np.uint8))


def resample(x, sr_to):
    return signal.resample_poly(x, sr_to, SR)

A = os.path.join(OUT, "audio")
save_wav(f"{A}/amostra_44k_16bit.wav", clean, SR)
save_wav(f"{A}/amostra_22k_16bit.wav", resample(clean, 22050), 22050)
save_wav(f"{A}/amostra_11k_8bit.wav", resample(clean, 11025), 11025, 8)
# 8 kHz sem filtro anti-aliasing -> aliasing audível (decimação ingênua)
naive = clean[:: int(SR / 8000)]
save_wav(f"{A}/amostra_8k_sem_filtro.wav", naive, SR // int(SR / 8000))
# 4 bits (quantização grosseira), guardado em 16 bits para tocar
q4 = np.round(clean * 7) / 7
save_wav(f"{A}/amostra_44k_4bit.wav", q4, SR)

# Formatos
def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)
src = f"{A}/amostra_44k_16bit.wav"
ff("-i", src, f"{A}/amostra.flac")
ff("-i", src, "-b:a", "192k", f"{A}/amostra_192k.mp3")
ff("-i", src, "-b:a", "32k", f"{A}/amostra_32k.mp3")

# ---------- 2. Antes x depois (edição real) ----------
t = np.arange(len(clean)) / SR
dirty = clean * 0.35                                          # gravação baixa
dirty += 0.04 * np.sin(2 * np.pi * 60 * t)                    # zumbido da rede (60 Hz)
dirty += 0.012 * np.sin(2 * np.pi * 180 * t)                  # harmônico do zumbido
dirty += 0.025 * rng.standard_normal(len(t))                  # chiado (ruído branco)
for c in rng.choice(len(t) - 50, 6, replace=False):            # cliques
    dirty[c:c + 3] += rng.choice([-1, 1]) * 0.9
save_wav(f"{A}/edicao_antes.wav", dirty, SR)

x = dirty.copy()
# (a) remoção de cliques: picos muito acima da mediana local
med = signal.medfilt(x, 9)
spk = np.abs(x - med) > 0.3
x[spk] = med[spk]
# (b) filtro passa-altas 100 Hz (remove o zumbido de 60/180 Hz)
sos = signal.butter(6, 100, "highpass", fs=SR, output="sos")
x = signal.sosfiltfilt(sos, x)
notch = signal.iirnotch(180, 20, SR)
x = signal.filtfilt(*notch, x)
# (c) redução de ruído por subtração espectral (perfil = silêncio inicial)
f, tt, Z = signal.stft(x, SR, nperseg=2048)
prof = np.mean(np.abs(Z[:, : int(LEAD * SR / 1024) - 2]), axis=1, keepdims=True)
mag = np.maximum(np.abs(Z) - 1.6 * prof, 0.06 * np.abs(Z))
_, x = signal.istft(mag * np.exp(1j * np.angle(Z)), SR, nperseg=2048)
x = x[: len(dirty)]
# (d) normalização para -1 dBFS e fade-out
x = x / np.max(np.abs(x)) * 0.89
x[-int(0.5 * SR):] *= np.linspace(1, 0, int(0.5 * SR))
save_wav(f"{A}/edicao_depois.wav", x, SR)

# ---------- 3. Efeitos do quiz ----------
def tone_seq(notes, dur=0.11, wave="sine"):
    out = []
    for n in notes:
        tt_ = np.arange(int(dur * SR)) / SR
        w = np.sin(2 * np.pi * note_hz(n) * tt_)
        if wave == "square":
            w = np.sign(w) * 0.5
        out.append(w * np.exp(-tt_ * 9))
    return np.concatenate(out) * 0.6
save_wav(f"{A}/sfx_acerto.wav", tone_seq([72, 76, 79, 84]), SR)
save_wav(f"{A}/sfx_erro.wav", tone_seq([52, 47], 0.22, "square"), SR)
save_wav(f"{A}/sfx_fim.wav", tone_seq([60, 64, 67, 72, 67, 72, 76], 0.13), SR)
save_wav(f"{A}/sfx_clique.wav", tone_seq([96], 0.04), SR)

# ---------- 4. Música MIDI composta (trilha do menu) ----------
def vlq(n):
    b = [n & 0x7F]
    n >>= 7
    while n:
        b.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(b)

TPQ = 480
def track(events):  # events: (tick, bytes)
    events.sort(key=lambda e: (e[0], e[1][0] & 0xF0 == 0x90))
    data, last = b"", 0
    for tk, ev in events:
        data += vlq(tk - last) + ev
        last = tk
    data += b"\x00\xff\x2f\x00"
    return b"MTrk" + struct.pack(">I", len(data)) + data

def notes_to_ev(notes, ch, vel):
    ev = []
    for n, s, d in notes:
        ev.append((int(s * TPQ), bytes([0x90 | ch, n, vel])))
        ev.append((int((s + d) * TPQ) - 10, bytes([0x80 | ch, n, 0])))
    return ev

chords = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]  # Am F C G
lead = [76, 72, 69, 72, 77, 76, 72, 69, 72, 69, 67, 64, 67, 71, 74, 71]
mel, bas, pad = [], [], []
for bar in range(8):
    ch_ = chords[bar % 4]
    base = bar * 4
    for k in range(4):
        pad += [(ch_[(k) % 3] + 12, base + k, 0.5), (ch_[(k + 1) % 3] + 12, base + k + 0.5, 0.5)]
    bas += [(ch_[0] - 12, base, 1.5), (ch_[0] - 12, base + 2, 1), (ch_[0] - 5, base + 3, 1)]
    if bar >= 2:
        i = (bar % 4) * 4
        mel += [(lead[i], base, 1), (lead[i + 1], base + 1, 0.5), (lead[i + 2], base + 1.5, 1.5), (lead[i + 3], base + 3, 1)]
tempo = 60_000_000 // 104
t0 = track([(0, b"\xff\x51\x03" + tempo.to_bytes(3, "big")),
            (0, b"\xff\x03" + vlq(19) + b"DggSounds - Trilha "[:19]),
            (0, b"\xff\x58\x04\x04\x02\x18\x08")])
t1 = track([(0, b"\xc0\x51")] + notes_to_ev(mel, 0, 96))    # lead
t2 = track([(0, b"\xc1\x04")] + notes_to_ev(pad, 1, 64))    # arpejo
t3 = track([(0, b"\xc2\x26")] + notes_to_ev(bas, 2, 100))   # baixo
mid = b"MThd" + struct.pack(">IHHH", 6, 1, 4, TPQ) + t0 + t1 + t2 + t3
with open(f"{OUT}/midi/trilha_menu.mid", "wb") as fh:
    fh.write(mid)

# renderização da trilha em WAV (para comparar tamanho MIDI x áudio)
secs = 32 * 60 / 104 + 1
song = np.zeros(int(secs * SR))
for notes, amp, kind in ((mel, 0.25, "sq"), (pad, 0.12, "tri"), (bas, 0.3, "saw")):
    for n, s, d in notes:
        dd = d * 60 / 104
        tt_ = np.arange(int((dd + 0.15) * SR)) / SR
        ph = note_hz(n) * tt_
        w = {"sq": np.sign(np.sin(2 * np.pi * ph)) * 0.6,
             "tri": 2 * np.abs(2 * (ph % 1) - 1) - 1,
             "saw": 2 * (ph % 1) - 1}[kind]
        env = np.minimum(1, tt_ / 0.01) * np.where(tt_ < dd, 1, np.exp(-(tt_ - dd) * 30))
        i = int(s * 60 / 104 * SR)
        seg = w * env * amp
        song[i:i + len(seg)] += seg[: len(song) - i]
song /= np.max(np.abs(song)) * 1.1
save_wav(f"{A}/trilha_renderizada.wav", song, SR)

# ---------- 5. Imagens matriciais (PNG): forma de onda e espectrograma ----------
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def cmap(v):  # 0..1 -> RGB (preto -> vermelho -> amarelo -> branco)
    stops = np.array([[22, 22, 22], [70, 70, 72], [140, 138, 134], [205, 203, 198], [255, 255, 255]])
    p = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.floor(p).astype(int).clip(0, len(stops) - 2)
    fr = (p - i)[..., None]
    return (stops[i] * (1 - fr) + stops[i + 1] * fr).astype(np.uint8)

def spectro_png(x, path, title):
    f_, t_, S = signal.spectrogram(x, SR, nperseg=1024, noverlap=768)
    S = 10 * np.log10(S + 1e-12)
    keep = f_ <= 8000
    S = S[keep][::-1]
    S = (S - (S.max() - 60)) / 60
    img = Image.fromarray(cmap(S)).resize((900, 300), Image.BICUBIC)
    canvas = Image.new("RGB", (960, 360), (22, 22, 22))
    canvas.paste(img, (50, 30))
    d = ImageDraw.Draw(canvas)
    fnt = ImageFont.truetype(FONT, 13)
    d.text((50, 8), title, fill=(255, 255, 255), font=ImageFont.truetype(FONTB, 15))
    for k, hz in enumerate([8000, 6000, 4000, 2000, 0]):
        y = 30 + int(300 * (1 - hz / 8000))
        d.text((4, min(y - 7, 322)), f"{hz//1000}k", fill=(200, 200, 210), font=fnt)
    d.text((50, 336), "tempo (s) →   frequência (Hz) ↑   cor = intensidade (dB)", fill=(200, 200, 210), font=fnt)
    canvas.save(path)

def wave_png(x, path, title, color):
    W, H = 960, 240
    im = Image.new("RGB", (W, H), (236, 234, 230))
    d = ImageDraw.Draw(im)
    d.line([(40, H // 2 + 10), (W - 10, H // 2 + 10)], fill=(200, 200, 200))
    cols = W - 50
    chunk = len(x) // cols
    for c in range(cols):
        seg = x[c * chunk:(c + 1) * chunk]
        y1 = H // 2 + 10 - int(seg.max() * 95)
        y2 = H // 2 + 10 - int(seg.min() * 95)
        d.line([(40 + c, y1), (40 + c, y2)], fill=color)
    d.text((40, 6), title, fill=(27, 27, 34), font=ImageFont.truetype(FONTB, 15))
    for v, lab in ((1, "+1"), (0, "0"), (-1, "−1")):
        d.text((6, H // 2 + 3 - int(v * 95)), lab, fill=(110, 110, 120), font=ImageFont.truetype(FONT, 12))
    im.save(path)

I = os.path.join(OUT, "img")
spectro_png(dirty, f"{I}/espectro_antes.png", "Antes: chiado em todas as frequências + zumbido de 60 Hz")
spectro_png(x, f"{I}/espectro_depois.png", "Depois: ruído reduzido, zumbido removido")
wave_png(dirty, f"{I}/onda_antes.png", "Antes: volume baixo, cliques (picos) e ruído de fundo", (140, 138, 134))
wave_png(x, f"{I}/onda_depois.png", "Depois: cliques removidos, normalizado a −1 dBFS", (26, 26, 26))

# ---------- 6. Dados embutidos para o site (funciona via file://) ----------
sizes = {fn: os.path.getsize(os.path.join(A, fn)) for fn in os.listdir(A)}
sizes["trilha_menu.mid"] = len(mid)
with open(f"{OUT}/js/dados.js", "w") as fh:
    fh.write("// gerado por tools/gen_audio.py\n")
    fh.write("window.DGG_MIDI_B64 = '" + base64.b64encode(mid).decode() + "';\n")
    bat = open(f"{OUT}/midi/bateria.mid", "rb").read()  # composta pelo grupo no LMMS
    sizes["bateria.mid"] = len(bat)
    fh.write("window.DGG_BATERIA_B64 = '" + base64.b64encode(bat).decode() + "';\n")
    fh.write("window.DGG_TAMANHOS = " + json.dumps(sizes, indent=1) + ";\n")
np.save("/home/claude/tools/_clean.npy", clean)
np.save("/home/claude/tools/_dirty.npy", dirty)
np.save("/home/claude/tools/_fixed.npy", x)
print(json.dumps(sizes, indent=1))
