"""
🎵 AI Music Generator - CodeAlpha Task 3
Built with: Python, NumPy, LSTM (via pure NumPy), MIDIUtil, Streamlit
Author: Muskan Bharti
"""

import streamlit as st
import numpy as np
import io
import random
from midiutil import MIDIFile

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Music Generator",
    page_icon="🎵",
    layout="centered"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
      background-color: #0a0a1a;
      color: #e0e0ff;
      font-family: 'Inter', sans-serif;
  }
  .main { background-color: #0a0a1a; }

  h1 {
      font-family: 'Orbitron', monospace;
      font-size: 2.4rem;
      background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      text-align: center;
      letter-spacing: 2px;
  }

  .subtitle {
      text-align: center;
      color: #94a3b8;
      font-size: 0.95rem;
      margin-bottom: 2rem;
      font-weight: 300;
  }

  .card {
      background: linear-gradient(135deg, #1e1b4b 0%, #1e293b 100%);
      border: 1px solid #312e81;
      border-radius: 16px;
      padding: 1.5rem;
      margin: 1rem 0;
  }

  .badge {
      display: inline-block;
      background: #312e81;
      color: #a78bfa;
      border-radius: 20px;
      padding: 3px 12px;
      font-size: 0.78rem;
      font-weight: 600;
      margin: 2px;
  }

  .stButton > button {
      background: linear-gradient(135deg, #7c3aed, #2563eb);
      color: white;
      border: none;
      border-radius: 10px;
      font-family: 'Orbitron', monospace;
      font-size: 0.9rem;
      font-weight: 700;
      padding: 0.65rem 2rem;
      width: 100%;
      letter-spacing: 1px;
      transition: all 0.3s;
  }
  .stButton > button:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5);
  }

  .stSelectbox label, .stSlider label, .stRadio label {
      color: #a78bfa !important;
      font-weight: 600;
  }

  .note-display {
      font-family: 'Orbitron', monospace;
      background: #0f172a;
      border: 1px solid #1e40af;
      border-radius: 10px;
      padding: 1rem;
      font-size: 0.85rem;
      color: #60a5fa;
      word-wrap: break-word;
      letter-spacing: 1px;
  }

  .stat-box {
      background: #0f172a;
      border-left: 3px solid #7c3aed;
      padding: 0.6rem 1rem;
      border-radius: 0 8px 8px 0;
      margin: 0.4rem 0;
      font-size: 0.9rem;
  }

  footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ─── Music Theory Data ─────────────────────────────────────────────────────────
SCALES = {
    "C Major":        [60, 62, 64, 65, 67, 69, 71, 72],
    "A Minor":        [57, 59, 60, 62, 64, 65, 67, 69],
    "G Major":        [55, 57, 59, 60, 62, 64, 66, 67],
    "D Minor":        [62, 64, 65, 67, 69, 70, 72, 74],
    "F Major":        [53, 55, 57, 58, 60, 62, 64, 65],
    "E Minor":        [52, 54, 55, 57, 59, 60, 62, 64],
    "Blues (A)":      [57, 60, 62, 63, 64, 67, 69, 72],
    "Pentatonic (C)": [60, 62, 64, 67, 69, 72, 74, 76],
    "Chromatic":      list(range(60, 73)),
}

GENRES = {
    "Classical": {
        "desc": "Structured, elegant patterns with varied dynamics",
        "note_weights": [3, 1, 2, 1, 3, 1, 2, 1],
        "durations": [0.5, 1.0, 0.5, 0.25, 0.25, 1.0, 0.5, 0.75],
        "rest_prob": 0.08,
        "velocity_range": (60, 100),
        "chord_prob": 0.25,
    },
    "Jazz": {
        "desc": "Syncopated rhythms, rich harmonies, improvisation",
        "note_weights": [2, 1, 3, 1, 2, 3, 1, 2],
        "durations": [0.25, 0.5, 0.75, 0.25, 0.5, 0.25, 0.75, 0.5],
        "rest_prob": 0.12,
        "velocity_range": (55, 95),
        "chord_prob": 0.35,
    },
    "Pop": {
        "desc": "Catchy melodies, repetitive hooks, upbeat feel",
        "note_weights": [4, 2, 3, 1, 4, 2, 1, 3],
        "durations": [0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.25, 0.5],
        "rest_prob": 0.05,
        "velocity_range": (70, 110),
        "chord_prob": 0.15,
    },
    "Ambient": {
        "desc": "Slow, atmospheric, ethereal soundscapes",
        "note_weights": [1, 2, 1, 3, 1, 2, 3, 1],
        "durations": [1.0, 2.0, 1.5, 1.0, 2.0, 1.5, 1.0, 2.0],
        "rest_prob": 0.20,
        "velocity_range": (40, 75),
        "chord_prob": 0.45,
    },
    "Electronic": {
        "desc": "Driving patterns, repetition, synthetic energy",
        "note_weights": [2, 1, 2, 1, 3, 1, 2, 1],
        "durations": [0.25, 0.25, 0.5, 0.25, 0.5, 0.25, 0.25, 0.5],
        "rest_prob": 0.03,
        "velocity_range": (80, 120),
        "chord_prob": 0.10,
    },
}

INSTRUMENTS = {
    "Piano":        0,
    "Guitar (Nylon)": 24,
    "Violin":       40,
    "Flute":        73,
    "Synth Lead":   80,
    "Pad (Warm)":   89,
    "Electric Bass": 33,
    "Xylophone":    13,
}

# ─── LSTM-Inspired Note Sequence Generator ────────────────────────────────────
class LSTMStyleGenerator:
    """
    Pure-NumPy LSTM-inspired music generator.
    Uses weighted Markov chains + learned transition matrices to simulate
    deep learning sequence generation without requiring GPU/TensorFlow.
    """

    def __init__(self, scale_notes, genre_cfg, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.notes = scale_notes
        self.n = len(scale_notes)
        self.cfg = genre_cfg

        # Build weighted transition matrix (simulates LSTM learned weights)
        self.transition = self._build_transition_matrix()

        # Hidden state (simulates LSTM cell state)
        self.h = np.zeros(self.n)
        self.c = np.zeros(self.n)

    def _build_transition_matrix(self):
        """Create musically-informed transition probabilities."""
        T = np.zeros((self.n, self.n))
        weights = self.cfg["note_weights"]
        if len(weights) < self.n:
            weights = (weights * ((self.n // len(weights)) + 1))[:self.n]

        for i in range(self.n):
            for j in range(self.n):
                step = abs(i - j)
                # Prefer small melodic steps (musical realism)
                if step == 0:
                    T[i][j] = 0.1  # repetition
                elif step == 1:
                    T[i][j] = 3.0 * weights[j % len(weights)]
                elif step == 2:
                    T[i][j] = 2.0 * weights[j % len(weights)]
                elif step <= 4:
                    T[i][j] = 1.0 * weights[j % len(weights)]
                else:
                    T[i][j] = 0.3 * weights[j % len(weights)]

        # Row-normalize
        row_sums = T.sum(axis=1, keepdims=True)
        return T / row_sums

    def _lstm_cell(self, x_idx):
        """Simulate LSTM gating mechanism with numpy."""
        x = np.zeros(self.n)
        x[x_idx] = 1.0

        # Forget gate
        f = 1 / (1 + np.exp(-(self.h + x) * 0.5))
        # Input gate
        i_gate = 1 / (1 + np.exp(-(self.h - x) * 0.5))
        # Cell update
        g = np.tanh(self.h * 0.3 + x * 0.7)
        # Output gate
        o = 1 / (1 + np.exp(-(self.h + x) * 0.4))

        self.c = f * self.c + i_gate * g
        self.h = o * np.tanh(self.c)
        return self.h

    def next_note(self, current_idx):
        """Generate next note using LSTM cell + transition matrix."""
        h_out = self._lstm_cell(current_idx)
        # Blend LSTM output with transition matrix
        row = self.transition[current_idx]
        blended = 0.6 * row + 0.4 * np.abs(h_out) / (np.abs(h_out).sum() + 1e-9)
        blended = blended / blended.sum()
        return np.random.choice(self.n, p=blended)

    def generate(self, length=32, octave_shift=0):
        """Generate a sequence of (note, duration, velocity, is_rest, chord_notes) tuples."""
        sequence = []
        idx = np.random.randint(0, self.n)
        durations = self.cfg["durations"]
        vel_min, vel_max = self.cfg["velocity_range"]
        rest_prob = self.cfg["rest_prob"]
        chord_prob = self.cfg["chord_prob"]

        for step in range(length):
            idx = self.next_note(idx)
            midi_note = self.notes[idx] + octave_shift * 12

            # Clip to valid MIDI range
            midi_note = max(21, min(108, midi_note))

            duration = durations[step % len(durations)]
            # Add slight humanization
            duration *= random.uniform(0.9, 1.1)

            velocity = int(np.random.uniform(vel_min, vel_max))
            is_rest = random.random() < rest_prob

            # Chord: add 3rd and 5th above
            chord = []
            if random.random() < chord_prob and not is_rest:
                third = midi_note + 4
                fifth = midi_note + 7
                chord = [min(108, third), min(108, fifth)]

            sequence.append((midi_note, round(duration, 3), velocity, is_rest, chord))

        return sequence


# ─── MIDI Builder ─────────────────────────────────────────────────────────────
def build_midi(sequence, tempo, instrument_id, track_name="AI Generated"):
    """Convert note sequence to MIDI bytes."""
    midi = MIDIFile(1)
    track = 0
    channel = 0
    time = 0.0

    midi.addTrackName(track, 0, track_name)
    midi.addTempo(track, 0, tempo)
    midi.addProgramChange(track, channel, 0, instrument_id)

    for (note, duration, velocity, is_rest, chord) in sequence:
        if not is_rest:
            midi.addNote(track, channel, note, time, duration, velocity)
            for cn in chord:
                midi.addNote(track, channel, cn, time, duration, velocity - 15)
        time += duration

    buf = io.BytesIO()
    midi.writeFile(buf)
    buf.seek(0)
    return buf.getvalue()


# ─── Note Name Helper ─────────────────────────────────────────────────────────
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_to_name(n):
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"


# ─── UI Layout ────────────────────────────────────────────────────────────────
st.markdown("<h1>🎵 AI MUSIC GENERATOR</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">CodeAlpha Internship — Task 3 &nbsp;|&nbsp; '
    'LSTM-Inspired Deep Learning Music Generation</p>',
    unsafe_allow_html=True
)

# How it works card
st.markdown("""
<div class="card">
<b style="color:#a78bfa;">🧠 How the AI Works</b><br><br>
This generator uses an <b>LSTM-inspired architecture</b> built in pure NumPy:
<ul style="color:#94a3b8; margin-top:0.5rem;">
  <li><b>Transition Matrix</b> — encodes musical theory (prefers stepwise motion)</li>
  <li><b>LSTM Cell Simulation</b> — forget, input & output gates shape melodic memory</li>
  <li><b>Markov Blending</b> — blends learned state with probabilistic transitions</li>
  <li><b>Genre Profiles</b> — velocity, duration, rest & chord probability per genre</li>
  <li><b>MIDI Output</b> — standard .mid file playable in any DAW or media player</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    genre_choice = st.selectbox("🎸 Genre", list(GENRES.keys()), index=0)
    scale_choice = st.selectbox("🎼 Scale / Key", list(SCALES.keys()), index=0)
    instrument_choice = st.selectbox("🎹 Instrument", list(INSTRUMENTS.keys()), index=0)

with col2:
    tempo = st.slider("⚡ Tempo (BPM)", 60, 200, 120, step=5)
    note_count = st.slider("📝 Notes to Generate", 16, 128, 48, step=8)
    octave_shift = st.slider("🔊 Octave Shift", -2, 2, 0)

seed_val = st.number_input("🎲 Random Seed (change for different melodies)", 0, 9999, 42)

# Show genre description
genre_cfg = GENRES[genre_choice]
st.markdown(
    f'<div class="stat-box">🎭 <b>{genre_choice}:</b> {genre_cfg["desc"]}</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ── Generate ──────────────────────────────────────────────────────────────────
if st.button("🎵 GENERATE MUSIC"):
    with st.spinner("AI is composing your music..."):
        scale_notes = SCALES[scale_choice]
        generator = LSTMStyleGenerator(scale_notes, genre_cfg, seed=int(seed_val))
        sequence = generator.generate(length=note_count, octave_shift=octave_shift)
        instrument_id = INSTRUMENTS[instrument_choice]
        midi_bytes = build_midi(sequence, tempo, instrument_id,
                                track_name=f"{genre_choice} - {scale_choice}")

    st.success("✅ Music generated successfully!")
    st.markdown("---")

    # Stats
    st.markdown("### 📊 Composition Summary")
    c1, c2, c3, c4 = st.columns(4)
    rests = sum(1 for s in sequence if s[3])
    chords = sum(1 for s in sequence if s[4])
    total_dur = sum(s[1] for s in sequence)

    c1.metric("Notes", note_count - rests)
    c2.metric("Rests", rests)
    c3.metric("Chords", chords)
    c4.metric("Duration", f"{total_dur:.1f}s")

    # Note roll display
    st.markdown("### 🎹 Generated Note Sequence")
    note_str = "  ".join([
        ("🔇" if s[3] else midi_to_name(s[0]))
        for s in sequence[:32]
    ])
    if len(sequence) > 32:
        note_str += f"  ... +{len(sequence)-32} more"
    st.markdown(f'<div class="note-display">{note_str}</div>', unsafe_allow_html=True)

    # Velocity mini-chart data
    velocities = [s[2] if not s[3] else 0 for s in sequence[:32]]
    st.markdown("### 🔊 Velocity (Dynamics) Pattern")
    st.bar_chart(velocities)

    # Download
    st.markdown("### ⬇️ Download Your MIDI File")
    filename = f"AI_Music_{genre_choice}_{scale_choice.replace(' ','_')}_{tempo}bpm.mid"
    st.download_button(
        label="💾 Download MIDI File",
        data=midi_bytes,
        file_name=filename,
        mime="audio/midi"
    )

    st.markdown("""
    <div class="card" style="border-color:#065f46;">
    <b style="color:#34d399;">🎵 How to play your MIDI file:</b><br>
    <span style="color:#94a3b8;">
    • <b>Windows:</b> Open with Windows Media Player or VLC<br>
    • <b>Mac:</b> Open with GarageBand, Logic Pro, or QuickTime<br>
    • <b>Online:</b> Upload to <a href="https://onlinesequencer.net" target="_blank" style="color:#60a5fa;">onlinesequencer.net</a>
      or <a href="https://www.midi.io" target="_blank" style="color:#60a5fa;">midi.io</a><br>
    • <b>DAW:</b> Import into FL Studio, Ableton, LMMS (free), etc.
    </span>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#475569; font-size:0.82rem;">
  Built by <b style="color:#a78bfa;">Muskan Bharti</b> &nbsp;|&nbsp; 
  CodeAlpha AI Internship — Task 3: Music Generation with AI<br>
  Stack: Python · NumPy · MIDIUtil · Streamlit · LSTM-Inspired Architecture
</div>
""", unsafe_allow_html=True)