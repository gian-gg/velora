# Velora

A CLI that turns EPUB chapters into audiobook files using local TTS models.

## How it works

1. **Read** the EPUB and split it into chapters, in reading order.
2. **Normalize** the text for speech: numbers, dates, money, abbreviations and symbols become words.
3. **Chunk** it into sentence-sized pieces the model can handle.
4. **Synthesize** each chunk with a TTS model (Kokoro by default), running on your CPU.
5. **Encode** the audio with ffmpeg into a tagged `.m4a`, `.mp3` or `.wav`.

## Install

Needs [uv](https://docs.astral.sh/uv/) and `ffmpeg`.

```bash
uv tool install --editable ".[kokoro]"   # installs the `velora` command
velora models download kokoro            # one-time, ~350 MB
```

## Usage

```bash
velora inspect book.epub              # list chapters with word counts and estimated minutes
velora inspect book.epub --show 4     # preview chapter 4's text as it will be spoken

velora read book.epub --chapter 4               # -> book-ch04.m4a
velora read book.epub --chapters 4-6            # one file for chapters 4 to 6
velora read book.epub --chapters 4-6 --split    # one file per chapter
velora read book.epub --chapter 4 --voice bf_emma --speed 0.9 -o ch4.mp3
```

Chapter numbers are the `#` column from `velora inspect`, not the book's own numbering.

## Configure the voice

```bash
velora models voices kokoro                             # list voices
velora models config kokoro                             # show all settings
velora models config kokoro set voice af_heart:0.8,af_bella:0.2
velora models config kokoro set speed 0.9
velora models config kokoro unset speed                 # back to default
```

Settings are saved in `~/.config/velora/config.toml`. Flags on `velora read` override them for one run.

| setting | what it changes |
|---|---|
| `voice` | a preset, or a weighted blend of presets |
| `speed` | speaking rate (0.5 to 2) |
| `lang` | pronunciation: `auto`, `en-us` or `en-gb` |
| `sentence_pause`, `clause_pause` | pauses inside a chunk |
| `chunk_pause`, `paragraph_pause` | pauses between chunks and paragraphs |

## Development

```bash
uv sync
uv run pytest
uv run pre-commit install
```
