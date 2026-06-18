# 🎵 AI Music Generator 

> **Music Generation with AI** using LSTM-inspired architecture, built with Python & Streamlit

---

## 📌 Project Overview

This project implements **AI-driven music generation** for Artificial Intelligence Internship (Task 3). It uses a custom **LSTM-inspired deep learning architecture** built entirely in NumPy to generate unique MIDI music sequences across multiple genres and musical scales — all without requiring GPU or heavy ML frameworks.

---

## 🧠 AI Architecture

### LSTM-Inspired Generator (Pure NumPy)

The core model simulates how an LSTM neural network learns music patterns:

| Component | Description |
|-----------|-------------|
| **Transition Matrix** | Musically-weighted note-to-note probabilities (prefers stepwise motion) |
| **Forget Gate** | Sigmoid-based dampening of previous hidden state |
| **Input Gate** | Controls how much new information enters the cell |
| **Output Gate** | Shapes what part of cell state becomes the output |
| **Markov Blending** | Combines learned LSTM hidden state with stochastic transitions |

```
Input Note → LSTM Cell → Hidden State → Blend with Transition Matrix → Next Note
```

### Why NumPy Instead of TensorFlow?
This approach makes the project **dependency-light** and **cross-platform**, while still demonstrating the key concepts of LSTM sequence learning (gating, memory cells, weighted transitions). A full TensorFlow implementation is included as an extension in `lstm_train.py`.

---

## ✨ Features

- 🎸 **5 Genre Profiles** — Classical, Jazz, Pop, Ambient, Electronic
- 🎼 **9 Scales/Keys** — Major, Minor, Blues, Pentatonic, Chromatic
- 🎹 **8 Instruments** — Piano, Guitar, Violin, Flute, Synth, Bass & more
- ⚡ **Adjustable Tempo** — 60–200 BPM
- 📝 **Variable Length** — 16 to 128 notes
- 🎵 **Chord Generation** — Adds 3rds and 5ths for harmonic richness
- 🔇 **Rest Probabilities** — Genre-specific silence modeling
- 🔊 **Dynamic Velocity** — Human-like expression per genre
- 💾 **MIDI Export** — Downloadable .mid file for any DAW or media player
- 📊 **Visual Analysis** — Note roll, dynamics chart, composition stats

---

## 🚀 How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Launch App
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 📂 Project Structure

```
CodeAlpha_MusicGenerationAI/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── samples/            # Sample generated MIDI files
    ├── classical_c_major.mid
    ├── jazz_blues_a.mid
    └── ambient_e_minor.mid
```

---

## 🎯 How to Use

1. **Select a Genre** — Each genre has unique rhythm, dynamics, and chord patterns
2. **Choose a Scale** — Sets the musical key and note palette
3. **Pick an Instrument** — Changes the MIDI program/patch
4. **Set Tempo** — BPM controls overall speed
5. **Adjust Note Count** — More notes = longer composition
6. **Change Seed** — Different seeds create entirely different melodies
7. **Click Generate** — AI composes your music instantly
8. **Download MIDI** — Open in any DAW, VLC, GarageBand, or online player

---

## 🎵 Playing Your MIDI File

| Platform | How to Play |
|----------|-------------|
| Windows  | VLC, Windows Media Player |
| Mac      | GarageBand, Logic Pro, QuickTime |
| Linux    | VLC, TiMidity++ |
| Online   | [onlinesequencer.net](https://onlinesequencer.net), [midi.io](https://midi.io) |
| DAW      | FL Studio, Ableton Live, LMMS (free) |

---

## 📊 Technical Details

### Genre Profiles

| Genre | Tempo Feel | Chord Prob | Rest Prob | Velocity Range |
|-------|-----------|------------|-----------|----------------|
| Classical | Varied | 25% | 8% | 60–100 |
| Jazz | Syncopated | 35% | 12% | 55–95 |
| Pop | Upbeat | 15% | 5% | 70–110 |
| Ambient | Slow | 45% | 20% | 40–75 |
| Electronic | Driving | 10% | 3% | 80–120 |

### MIDI Standards Used
- Format: Type 0 (single track)
- Channel: 0 (melodic)
- Program Change: GM instrument mapping
- Note range: MIDI 21–108 (piano range)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `Python 3.10+` | Core language |
| `NumPy` | LSTM simulation, math ops |
| `MIDIUtil` | MIDI file generation |
| `Streamlit` | Web UI framework |

---

## 👩‍💻 Author

**Muskan Bharti**  
B.Tech CSE (Networks), GEC Kaimur — Batch 2023–2027  
🔗 [LinkedIn](www.linkedin.com/in/muskanbhartii) | 🐙 [GitHub](https://github.com/muskanbharti01)


## 📄 License

MIT License — Free to use and modify with attribution.
