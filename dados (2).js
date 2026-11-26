"""Gera a vídeo-aula (MP4 H.264 + AAC) quadro a quadro com PIL e ffmpeg.
Uso: python3 gen_video.py <pasta_site>
"""
import sys, os, subprocess, math
import numpy as np
from scipy.io import wavfile
from scipy import signal
from PIL import Image, ImageDraw, ImageFont

SITE = sys.argv[1]
W, H, FPS, SR = 960, 540, 24, 44100
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
f_big, f_mid, f_sm = ImageFont.truetype(FB, 44), ImageFont.truetype(FB, 26), ImageFont.truetype(FR, 21)
RED, INK, YEL, BLUE, PAPER = (26, 26, 26), (26, 26, 26), (236, 234, 230), (26, 26, 26), (236, 234, 230)

def load(name):
    sr, x = wavfile.read(os.path.join(SITE, "audio", name))
    x = (x.astype(float) - 128) / 127 if x.dtype == np.uint8 else x.astype(float) / 32767
    return signal.resample_poly(x, SR, sr) if sr != SR else x

def fit(x, secs, fade=0.3):
    n = int(secs * SR)
    x = np.pad(x, (0, max(0, n - len(x))))[:n].copy()
    f = int(fade * SR)
    x[:f] *= np.linspace(0, 1, f); x[-f:] *= np.linspace(1, 0, f)
    return x

SCENES = [  # (duração, legenda, trilha)
    (5, "", fit(load("trilha_renderizada.wav"), 5) * 0.7),
    (9, "O microfone transforma a vibração do ar em um sinal elétrico contínuo.", fit(load("amostra_44k_16bit.wav"), 9)),
    (10, "Amostragem: medir o sinal muitas vezes por segundo — 44.100 vezes no CD.", fit(load("amostra_44k_16bit.wav")[SR:], 10)),
    (9, "Com poucas amostras e sem filtro, surgem distorções: é o aliasing.", fit(load("amostra_8k_sem_filtro.wav"), 9)),
    (9, "Quantização: cada amostra vira um número inteiro. Menos bits, mais ruído.", fit(load("amostra_44k_4bit.wav"), 9)),
    (10, "Na edição removemos cliques, zumbido e chiado, e normalizamos o volume.",
     np.concatenate([fit(load("edicao_antes.wav"), 5), fit(load("edicao_depois.wav"), 5)])),
    (5, "", fit(load("sfx_fim.wav"), 5, 0.05)),
]

LOGO = [  # mesmos traços de svg/logo.svg (viewBox 40 30 915 230)
    [(60,45),(180,133),(60,240),(60,45)],
    [(292,65),(236,65),(218,84),(218,180),(240,197),(286,197),(312,162),(312,145),(268,145),(268,166)], [(292,65),(310,84)],
    [(414,65),(358,65),(340,84),(340,180),(362,197),(408,197),(434,162),(434,145),(390,145),(390,166)], [(414,65),(432,84)],
    [(540,68),(455,68),(455,130),(540,130),(540,195),(455,195)],
    [(570,70),(645,70),(645,195),(570,195),(570,70)],
    [(668,70),(668,195),(740,195),(740,70)],
    [(765,195),(765,70),(800,195),(800,70),(860,135),(800,195)],
    [(935,70),(872,70),(872,130),(935,130),(935,195),(872,195)],
]
def logo(d, x0, y0, k, cor):
    for pl in LOGO:
        d.line([(x0 + (x - 40) * k, y0 + (y - 30) * k) for x, y in pl], fill=cor, width=max(2, int(12 * k)), joint="curve")

def wrap(d, text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) > maxw and cur:
            lines.append(cur); cur = w
        else:
            cur = t
    return lines + [cur]

def caption(d, text):
    if not text: return
    lines = wrap(d, text, f_sm, W - 120)
    h = 22 + 28 * len(lines)
    d.rounded_rectangle([40, H - h - 24, W - 40, H - 24], 14, fill=INK)
    for i, ln in enumerate(lines):
        d.text((60, H - h - 13 + i * 28), ln, font=f_sm, fill=(255, 255, 255))

def wave_y(x, t, amp=110, cy=230):
    return cy - amp * (0.75 * math.sin(2 * math.pi * (x / 260 - t * 0.6)) + 0.25 * math.sin(2 * math.pi * (x / 87 - t * 1.3)))

def axes(d, cy=230):
    d.line([(60, cy), (W - 60, cy)], fill=(190, 188, 184), width=2)

def frame(si, t, T):
    im = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(im)
    dur, cap, _ = SCENES[si]
    p = t / dur
    if si == 0:
        d.rectangle([0, 0, W, H], fill=RED)
        for k in range(28):
            hgt = 30 + 120 * abs(math.sin(k * 0.7 + t * 4))
            d.rounded_rectangle([90 + k * 28, 300 - hgt / 2, 106 + k * 28, 300 + hgt / 2], 8, fill=(255, 255, 255))
        logo(d, 90, 50, 0.42, (255, 255, 255))
        d.text((90, 160), "Vídeo-aula: como o som vira número", font=f_mid, fill=YEL)
    elif si in (1, 2, 3):
        axes(d)
        pts = [(x, wave_y(x, t)) for x in range(60, W - 60, 3)]
        d.line(pts, fill=RED if si == 1 else (178, 176, 172), width=5 if si == 1 else 3)
        if si == 1:
            d.text((60, 40), "1. Sinal analógico (contínuo)", font=f_mid, fill=INK)
        else:
            step = int(70 - 50 * min(p * 1.5, 1)) if si == 2 else 64
            d.text((60, 40), "2. Amostragem" if si == 2 else "3. Taxa baixa demais", font=f_mid, fill=INK)
            prev = None
            for x in range(60, W - 60, step):
                y = wave_y(x, t)
                if si == 3 and prev:
                    d.line([prev, (x, prev[1]), (x, y)], fill=BLUE, width=3)
                d.line([(x, 230), (x, y)], fill=(150, 150, 150), width=1)
                d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=BLUE)
                prev = (x, y)
            lab = f"{len(range(60, W - 60, step))} amostras na tela" if si == 2 else "8.000 Hz, sem filtro"
            d.text((W - 60 - d.textlength(lab, font=f_mid), 44), lab, font=f_mid, fill=BLUE)
    elif si == 4:
        bits = max(2, 6 - int(p * 5))
        levels = 2 ** bits
        d.text((60, 40), "4. Quantização", font=f_mid, fill=INK)
        d.text((W - 340, 44), f"{bits} bits = {levels} níveis", font=f_mid, fill=BLUE)
        for k in range(levels + 1):
            y = 230 - 140 + 280 * k / levels
            d.line([(60, y), (W - 60, y)], fill=(214, 212, 207))
        pts = [(x, wave_y(x, 0.4)) for x in range(60, W - 60, 3)]
        d.line(pts, fill=(178, 176, 172), width=3)
        q = []
        for x in range(60, W - 60, 3):
            v = (wave_y(x, 0.4) - 90) / 280 * levels
            q.append((x, 90 + round(v) * 280 / levels))
        d.line(q, fill=BLUE, width=4)
    elif si == 5:
        d.text((60, 30), "5. Edição: antes → depois", font=f_mid, fill=INK)
        a = Image.open(os.path.join(SITE, "img", "onda_antes.png")).resize((840, 210))
        b = Image.open(os.path.join(SITE, "img", "onda_depois.png")).resize((840, 210))
        mix = min(max((t - 4.6) / 0.8, 0), 1)
        im.paste(Image.blend(a, b, mix), (60, 110))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([60, 340, 60 + 840 * p, 352], 6, fill=INK)
    elif si == 6:
        d.rectangle([0, 0, W, H], fill=INK)
        logo(d, 90, 60, 0.3, (200, 198, 194))
        d.text((90, 150), "Agora é com você!", font=f_big, fill=(255, 255, 255))
        d.text((90, 215), "Explore os módulos e faça o quiz.", font=f_mid, fill=YEL)
        d.text((90, 300), "Guilherme R. L. da Silva · Iago de O. Almeida", font=f_sm, fill=(220, 220, 228))
        d.text((90, 332), "José Roberto R. L. de Andrade · Pedro Moreno Vieira", font=f_sm, fill=(220, 220, 228))
    caption(d, cap)
    # barra de progresso global
    d.rectangle([0, H - 6, W * T, H], fill=(140, 140, 140))
    return im

total = sum(s[0] for s in SCENES)
audio = np.concatenate([s[2] for s in SCENES]) * 0.9
wavfile.write("/tmp/vid_audio.wav", SR, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
os.makedirs(os.path.join(SITE, "video"), exist_ok=True)
out = os.path.join(SITE, "video", "videoaula_amostragem.mp4")
proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", "/tmp/vid_audio.wav",
                         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                         "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", out],
                        stdin=subprocess.PIPE)
n = 0
for si, (dur, _, _) in enumerate(SCENES):
    for k in range(int(dur * FPS)):
        t = k / FPS
        im = frame(si, t, n / (total * FPS))
        proc.stdin.write(im.tobytes())
        if k == int(dur * FPS * 0.6):
            im.save(f"/tmp/vframe_{si}.png")
        n += 1
proc.stdin.close(); proc.wait()
# pôster do vídeo (imagem matricial)
frame(2, 3, 0.3).save(os.path.join(SITE, "img", "video_poster.jpg"), quality=85)
print(out, os.path.getsize(out))
