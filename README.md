# Ghid Complet: LLM Local cu llama.cpp

### Stack: llama.cpp · Qwen3-14B Claude 4.5 Opus Reasoning Distill · Linux · NVIDIA 8GB+ VRAM

---

## Cuprins

 [[#1. Ce este un LLM?]]
 [[#2. Specificații Hardware]]
 [[#3. Instalare llama.cpp]]
 [[#4. Descărcare Model GGUF]]
 [[#5. Pornire Server llama.cpp]]
 [[#6. Optimizare pentru coding]]
 [[#7. Troubleshooting]]
 [[#8. Modele Alternative]]
 [[#9. Cheat Sheet]]
 [[#10. Test]]

---

# 1. Ce este un LLM?

Un **Large Language Model (LLM)** este o rețea neuronală de tip **Transformer** antrenată pe cantități masive de text pentru a înțelege și genera limbaj natural și cod. La fiecare pas de inferență, modelul primește o secvență de **token-uri** și calculează probabilitatea pentru următorul token prin mecanismul de **Attention**.

Modelul folosit în acest ghid este **Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill**, un model Qwen3 de 14B parametri fine-tunat (distillat) pe un dataset generat de Claude Opus 4.5 cu efort înalt de raționament. Este specializat pe **coding, știință și raționament complex**, combinând arhitectura eficientă Qwen3 cu stilul de gândire al Claude Opus.

### 1.1. Quantizarea: cum încapi 14B parametri în 8GB VRAM

Un model de 14B parametri ocupă ~28 GB în FP32. Prin **quantizare** se reduce precizia numerică a fiecărui parametru, micșorând drastic dimensiunea cu pierdere minimă de calitate.

| Format     | Dimensiune | VRAM necesar | Calitate           |
| ---------- | ---------- | ------------ | ------------------ |
| BF16/F16   | 29.5 GB    | 30+ GB       | Referință          |
| Q8_0       | 15.7 GB    | 16+ GB       | Excelentă          |
| Q4_K_M     | 9.0 GB     | ~9.5 GB      | Foarte bună        |
| **IQ4_NL** | **8.6 GB** | **~8.8 GB**  | **Recomandat 8GB** |
| Q3_K_M     | 7.32 GB    | ~7.5 GB      | Bună, margine VRAM |
| Q3_K_S     | 6.66 GB    | ~7 GB        | Acceptabilă        |

> **IQ4_NL** este formatul recomandat pentru GPU-uri de 8 GB VRAM. Oferă calitate apropiată de Q4_K_M la o dimensiune mai mică. Dacă ai exact, cum am și eu, 8 GB și rulezi alte procese simultan, alege **Q3_K_M**.

### 1.2. De ce llama.cpp?

`llama.cpp` este motorul de inferență scris în C/C++ care stă la baza majorității tool-urilor LLM locale. Avantaje față de alternative:

- **Suport CUDA optim** — folosește cuBLAS pentru accelerație maximă pe NVIDIA
- **Format GGUF** — formatul universal pentru modele quantizate
- **Server HTTP** - foarte usor de folosit pentru a rula LLM-ul local
- **Control** — control total asupra configuratiei

---

## 2. Specificații Hardware

| Componentă | Minim Recomandat    | Rol                                 |
| ---------- | ------------------- | ----------------------------------- |
| GPU        | NVIDIA 8GB+ VRAM    | Inferență accelerată CUDA (cuBLAS)  |
| CPU        | 8+ core-uri moderne | Compilare llama.cpp, CPU offloading |
| RAM        | 16 GB DDR4+         | OS + layers offloaded din VRAM      |
| Driver     | NVIDIA 525+         | Necesar pentru CUDA 12.x            |

### Calculul VRAM

```
Model: Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill — format IQ4_NL

Dimensiune fișier GGUF : 8.6 GB
+ KV cache (ctx 8192)  : ~0.4 GB
+ overhead CUDA        : ~0.3 GB
= Total estimat        : ~9.3 GB

→ Pe GPU de 8 GB: voi folosi --n-gpu-layers 35 (offload ~2 layere pe RAM)
→ Pe GPU de 10+ GB: --n-gpu-layers 999 (totul pe GPU)
```

Alternativ cu **Q3_K_M (7.32 GB)** totul încape pe GPU de 8 GB fără offloading.

---

## 3. Instalare llama.cpp

### 3.1. Dependențe sistem

Verifică că CUDA este disponibil:

```bash
nvcc --version     # trebuie să afișeze CUDA 12.x
nvidia-smi         # trebuie să afișeze GPU-ul și driverul
```

### 3.2. Clonare și compilare cu suport CUDA

```bash
# Clonare repository
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Compilare cu CUDA (cuBLAS) — durează ~2-5 minute
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86   # Ampere (RTX 30xx)
  # Pentru RTX 40xx folosește: -DCMAKE_CUDA_ARCHITECTURES=89
  # Pentru RTX 20xx folosește: -DCMAKE_CUDA_ARCHITECTURES=75

cmake --build build --config Release -j$(nproc)
```

> **Architectures CUDA după generație:**
> 
> - RTX 20xx (Turing) → `75`
> - RTX 30xx (Ampere) → `86`
> - RTX 40xx (Ada) → `89`
> - RTX 50xx (Blackwell) → `120`

---

## 4. Descărcare Model GGUF

Modelul este disponibil pe [HuggingFace](https://huggingface.co/TeichAI/Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill-GGUF).

### 4.1. Alege formatul potrivit pentru VRAM-ul tău

| Format     | Dimensiune | Când să alegi                             |
| ---------- | ---------- | ----------------------------------------- |
| Q3_K_M     | 7.32 GB    | GPU exact 8 GB, vrei margine de siguranță |
| **IQ4_NL** | **8.6 GB** | **GPU 8 GB+**                             |
| Q4_K_M     | 9.0 GB     | GPU 10+ GB                                |
| Q8_0       | 15.7 GB    | GPU 16+ GB, calitate maximă               |

### 4.2. Descărcare cu llama cpp

```bash
pip install huggingface-hub --break-system-packages

# Creează directorul pentru model
mkdir -p ~/models/

# Descarcă IQ4_NL (~8.6 GB) — recomandat pentru 8GB VRAM
LLAMA_CACHE=~/models llama-server \
           -hf TeichAI/Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill-GGUF:IQ4_NL \
           --host 0.0.0.0 \
             --port 8080 \
             --n-gpu-layers 999 \
             --ctx-size 8192 \
             --threads 8 \
             --parallel 2
```
<img width="876" height="594" alt="image" src="https://github.com/user-attachments/assets/a005f5f0-c74c-4388-99f2-0e819cc312fe" />

---

## 5. Pornire Server llama.cpp

`llama-server` expune un API HTTP compatibil OpenAI pe portul `8080`. Open WebUI se conectează direct la acest endpoint.

### 5.1. Comandă de bază

```bash
llama-server \
  --model ~/models/qwen3-14b-reasoning/Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill-IQ4_NL.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --n-gpu-layers 999 \
  --ctx-size 8192 \
  --threads 8 \
  --parallel 2
```

<img width="876" height="594" alt="image" src="https://github.com/user-attachments/assets/73204b84-5150-4768-8239-4e9421a8bfa7" />


### 5.2. Explicația parametrilor

| Parametru            | Valoare        | Efect                                                    |
| -------------------- | -------------- | -------------------------------------------------------- |
| `--n-gpu-layers 999` | max layers     | Trimite toate layerele pe GPU (dacă VRAM permite)        |
| `--n-gpu-layers 33`  | layers parțial | Offload parțial — pentru GPU-uri de exact 8 GB           |
| `--ctx-size 8192`    | token context  | Cât context poate procesa simultan (mai mare = mai lent) |
| `--threads 8`        | CPU threads    | Thread-uri pentru layerr pe CPU și pre-procesare         |
| `--parallel 2`       | sesiuni        | Câte cereri simultane poate gestiona serverul            |
| `--batch-size 512`   | batch tokens   | Token-uri procesate în paralel — crește throughput       |

### 5.3. Comandă optimizată pentru 8 GB VRAM

```bash
# Cu IQ4_NL — poate necesita offload ~2 layers pe RAM
llama-server \
  --model ~/models/qwen3-14b-reasoning/Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill-IQ4_NL.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --n-gpu-layers 35 \     # ajustează: 999 dacă VRAM permite, 30-35 dacă OOM
  --ctx-size 8192 \
  --threads $(nproc) \
  --batch-size 512 \
  --parallel 1 \
  --log-disable

# Cu Q3_K_M — totul pe GPU fără offloading
llama-server \
  --model ~/models/qwen3-14b-reasoning/Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill-Q3_K_M.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --n-gpu-layers 999 \
  --ctx-size 8192 \
  --threads $(nproc) \
  --batch-size 512 \
  --parallel 1 
```

### 5.4. Verificare server pornit

```bash
# Verifică că serverul ascultă
curl http://localhost:8080/health
# Output: {"status":"ok"}
```

<img width="441" height="50" alt="image" src="https://github.com/user-attachments/assets/e536a94f-5a9d-4198-b049-4af236d91f15" />


```
# Listează modelele disponibile (endpoint compatibil OpenAI)
curl http://localhost:8080/v1/models | python3 -m json.tool
```

<img width="868" height="375" alt="image" src="https://github.com/user-attachments/assets/d0041ca6-cb1e-477c-a82d-40937a77ac50" />


### 5.5. Rulare ca serviciu systemd

Pentru a porni serverul automat la boot:

```bash
sudo nano /etc/systemd/system/llama-server.service
```

```ini
[Unit]
Description=llama.cpp Server
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
ExecStart=/usr/local/bin/llama-server \
  --model /home/YOUR_USERNAME/models/qwen3-14b-reasoning/Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill-IQ4_NL.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --n-gpu-layers 35 \
  --ctx-size 8192 \
  --parallel 1
Restart=always
RestartSec=5
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=multi-user.target
```

```bash
# Activare serviciu
sudo systemctl daemon-reload
sudo systemctl enable llama-server
sudo systemctl start llama-server

# Verificare
sudo systemctl status llama-server
journalctl -u llama-server -f    # log-uri live
```

---

## 6. Optimizare pentru coding

### 6.1. System Prompt recomandat

Acest model este distilat pe raționamentul lui Claude Opus 4.5 — funcționează cel mai bine cu instrucțiuni clare și cereri de explicații pas cu pas.

În WebUI → Settings → General → System Message:

```
You are an expert software engineer with strong reasoning capabilities.
When solving problems:
- Think step by step before writing code
- Explain your reasoning, then provide the implementation
- Point out edge cases, potential bugs, and security issues
- Suggest improvements and alternatives where relevant
- Write clean, well-commented, production-ready code

Default to Python unless specified. For shell scripts, target Linux/bash.
Respond in the same language as the user (RO/EN).
```

### 6.2. Parametri de generare pentru cod

|Parametru|Valoare|Rațiune|
|---|---|---|
|Temperature|0.1 – 0.3|Cod determinist, sintaxă corectă|
|Top-P|0.9|Diversitate controlată|
|Repeat Penalty|1.1|Evită repetiția de cod inutil|
|Max Tokens|2048|Răspunsuri complete fără tăiere|

> Temperature > 0.7 crește creativitatea dar introduce erori de sintaxă și logică în cod. Menține-l sub 0.3 pentru programare.

---

## 7. Troubleshooting

### Modelul rulează pe CPU (lent, <5 tok/s)

```bash
# Verifică layers GPU în output-ul de start al serverului
# Trebuie să apară: "llm_load_tensors: offloaded XX/XX layers to GPU"

# Dacă 0 layers sunt pe GPU → problema cu CUDA build
./build/bin/llama-server --version
# Verifică că în output apare: "BLAS = 1, GPU = CUDA"

# Recompilează asigurând că nvcc este în PATH
which nvcc && nvcc --version
cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)
```

### Out of Memory (CUDA OOM)

```bash
# Reduce numărul de layers pe GPU
# Încearcă valori descrescătoare: 35 → 30 → 25
llama-server \
  --model ~/models/.../model.gguf \
  --n-gpu-layers 30 \     # ← reduce până dispare eroarea
  ...

# Sau folosește un context mai mic
--ctx-size 4096   # în loc de 8192
```

### Viteză inferență scăzută

| Simptom       | Cauză probabilă      | Soluție                   |
| ------------- | -------------------- | ------------------------- |
| < 5 tok/s     | Rulează pe CPU       | Recompilează cu CUDA      |
| 5 – 15 tok/s  | Offload parțial      | Crește `--n-gpu-layers`   |
| 15 – 40 tok/s | Normal pentru 14B Q4 | Performanță acceptabilă ✓ |
| > 40 tok/s    | Configurare optimă   | Excelent ✓                |

---

## 8. Modele Alternative

| Model (GGUF)                                         | Dimensiune | Specializare               |
| ---------------------------------------------------- | ---------- | -------------------------- |
| `TeichAI/Qwen3-8B-Claude-4.5-Opus-...`               | ~5 GB      | Reasoning, GPU 6GB         |
| `TeichAI/Qwen3-4B-Thinking-2507-Claude-4.5-...`      | ~3 GB      | Reasoning rapid, GPU 4GB   |
| **`TeichAI/Qwen3-14B-Claude-4.5-Opus-...` (IQ4_NL)** | **8.6 GB** | **Recomandat**             |
| `Qwen/Qwen2.5-Coder-14B-Instruct` (Q4_K_M)           | ~8.7 GB    | Coding pur, fără reasoning |
| `deepseek-ai/DeepSeek-Coder-V2-Lite` (Q4)            | ~9 GB      | Coding + math              |
| `meta-llama/Llama-3.1-8B-Instruct` (Q4)              | ~4.7 GB    | General purpose            |

Toate modelele TeichAI din aceeași serie sunt disponibile la [huggingface.co/TeichAI](https://huggingface.co/TeichAI).

---

## 9. Cheat Sheet

```bash
# ── BUILD ─────────────────────────────────────────────────────
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j$(nproc)

# ── MODEL ─────────────────────────────────────────────────────
huggingface-cli download \
  TeichAI/Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill-GGUF \
  Qwen3-14B-Claude-4.5-Opus-High-Reasoning-Distill-IQ4_NL.gguf \
  --local-dir ~/models/qwen3-14b-reasoning

# ── SERVER ────────────────────────────────────────────────────
llama-server \
  -m ~/models/model.gguf \
  --host 0.0.0.0 --port 8080 \
  --n-gpu-layers 35 --ctx-size 8192

# ── STATUS ────────────────────────────────────────────────────
curl http://localhost:8080/health         # verifică server
curl http://localhost:8080/v1/models      # listează modele
nvidia-smi                                # utilizare GPU
journalctl -u llama-server -f             # log-uri live
docker logs -f open-webui                 # log-uri WebUI

# ── ACCES ─────────────────────────────────────────────────────
# llama.cpp   →  http://localhost:8080
```

## 10. Test

Interfața built-in llama.cpp funcționează — primul prompt a primit răspuns cu o viteză de **10 token/s**, ceea ce confirmă că modelul rulează corect pe GPU.

<img width="1280" height="720" alt="image" src="https://github.com/user-attachments/assets/ea40ba6b-d61e-45f1-9ca7-10cb7e492087" />

---
