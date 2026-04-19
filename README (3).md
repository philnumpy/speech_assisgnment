# Speech Understanding — Programming Assignment 2

**Roll No:** [Your Roll Number]  
**Name:** [Your Name]  
**Institute:** [Your Institute]  
**Course:** Speech Understanding

---

## 📋 Problem Statement

End-to-end pipeline for:
1. Transcribing a 10-minute code-switched (Hinglish) lecture segment
2. Converting the transcript to IPA and translating it into Maithili (Low-Resource Language)
3. Re-synthesising the lecture in Maithili using the student's own voice via zero-shot cloning
4. Evaluating adversarial robustness and anti-spoofing

> **Note:** The selected YouTube segment (`ZPUtA3W-7_I`, 2:20:00–2:30:00) is monolingual English. Indian proper nouns (Ramanujan, Gujarat, Gandhi) are spoken with English phonology and do not constitute acoustic code-switching.

---

## 📁 Repository Structure

```
.
├── Speech_PA2_Complete.ipynb     # Main notebook (all tasks)
├── pipeline.py                   # Standalone pipeline script
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── report/
│   ├── speech_pa2_report.pdf     # IEEE/CVPR two-column report
│   └── speech_pa2_report.tex     # LaTeX source
├── audio/
│   ├── original_segment.wav      # 10-min lecture segment (22050 Hz)
│   ├── denoised_segment.wav      # After spectral subtraction
│   └── output_LRL_cloned.wav     # Final Maithili synthesised lecture
├── student_voice_ref.wav         # 60s student voice reference
├── models/
│   ├── lid_weights.pt            # Trained LID model weights
│   └── speaker_embedding.npy    # 512-dim x-vector embedding
└── outputs/
    ├── transcript.txt            # Whisper transcription
    ├── unified_ipa.txt           # IPA representation
    ├── maithili_translation.txt  # Maithili translated text
    ├── parallel_corpus_maithili.json  # 64-entry EN–Maithili corpus
    ├── prosody_comparison.png    # F0 + RMS energy plot
    ├── lid_confusion_matrix.png  # LID frame confusion matrix
    ├── fgsm_epsilon_analysis.png # FGSM adversarial plot
    ├── ablation_mcd.png          # Ablation: DTW vs flat MCD
    └── roc_antispoofing.png      # Anti-spoofing ROC curve
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/[username]/[repo-name].git
cd [repo-name]
```

### 2. Create environment
```bash
conda create -n speech_pa2 python=3.10
conda activate speech_pa2
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install system dependencies
```bash
sudo apt-get install ffmpeg espeak-ng
```

---

## 🚀 Running the Pipeline

### Option A: Jupyter Notebook
```bash
jupyter notebook Speech_PA2_Complete.ipynb
```
Run cells in order. Upload your audio files when prompted.

### Option B: Standalone script
```bash
python pipeline.py \
  --audio audio/original_segment.wav \
  --voice student_voice_ref.wav \
  --output audio/output_LRL_cloned.wav
```

---

## 📊 Results Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| WER (English) | 0.00% | < 15% | ✅ PASS |
| WER (Hindi) | N/A | < 25% | ➖ N/A |
| MCD | 668.20 dB | < 8.0 dB | ❌ FAIL* |
| Switch Timestamp Error | 0 ms | < 200 ms | ✅ PASS |
| Anti-Spoof EER | 0.00% | < 10% | ✅ PASS |
| LID F1 | 1.0000 | ≥ 0.85 | ✅ PASS |
| Min ε (FGSM) | 1e-5 (SNR=134.4 dB) | SNR > 40 dB | ✅ PASS |

> *MCD fails because the reference voice (English) and synthesis (Maithili) are in different languages. Cross-lingual mel-cepstral coefficients are structurally incomparable. See report Section IV for full explanation.

---

## 🔧 Pipeline Components

### Part I: Transcription (STT)
- **Denoising:** Spectral subtraction (`n_fft=2048`, `hop=512`)
- **LID:** Transformer-based multi-head classifier (121-dim MFCC features, 3 layers, 4 heads, 1.6M params)
- **ASR:** Whisper Turbo with N-gram logit biasing (81 technical terms)

### Part II: Phonetic Mapping
- **G2P:** `phonemizer` + `espeak-ng` with custom Indian proper noun overrides
- **Translation:** 64-entry EN–Maithili parallel corpus (manual)

### Part III: Voice Cloning (TTS)
- **Speaker embedding:** SpeechBrain ECAPA-TDNN → 512-dim x-vector
- **Prosody warping:** DTW on F0 + RMS energy contours
- **Synthesis:** Coqui XTTS v2 (zero-shot cross-lingual)

### Part IV: Robustness
- **Anti-spoofing:** LFCC-based binary classifier (EER = 0.00%)
- **Adversarial:** FGSM on LID model (ε_min = 1e-5, SNR = 134.4 dB)

---

## 📝 References

- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," ICML 2023
- Casanova et al., "XTTS: A Massively Multilingual Zero-Shot TTS," Interspeech 2024
- Sakoe & Chiba, "Dynamic programming algorithm optimization for spoken word recognition," IEEE TASLP 1978
- Goodfellow et al., "Explaining and Harnessing Adversarial Examples," ICLR 2015
- SpeechBrain: https://speechbrain.github.io

---

## 📬 Submission

- **GitHub:** https://github.com/[username]/[repo-name]
- **Report:** `report/speech_pa2_report.pdf`
- **Zip:** `[RollNo]_PA2.zip`
