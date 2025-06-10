# Multi-Role Cognitive Engine (MRCE)

MRCE is a small experimental framework exploring multi-agent interaction with a local LLM. Each agent keeps its own memory and the system coordinates them via a simple finite state machine. A "Phase‑Crystal" memory stores modulated embeddings and a FECR vector evolves from the stored crystals.

## Repository layout

```
configs/          # YAML persona definitions
logs/             # CSV conversation logs
mrce/             # Core package (agents, memory, FECR utilities)
scripts/          # Demo entrypoints
tests/            # Pytest suite
```

Key modules:

- **`mrce/phasecrystal/memory.py`** – sliding window memory that scores resonance and promotes stable "crystals".
- **`mrce/fecr/`** – helpers for computing the FECR vector and modulating embeddings.
- **`mrce/llm/ollama_client.py`** – minimal wrapper for a local Ollama server providing chat completions and embeddings.
- **`mrce/mrcore/`** – agents, coordinator FSM and the `MRCEngine` that ties everything together.
- **`scripts/smoke_engine.py`** – interactive demo using `MRCEngine`.

Persona configurations live under `configs/personae/` and define system prompts for each expert agent.

## Quick start

1. **Install dependencies** (Python 3.11):
   ```bash
   pip install -r requirements.txt
   ```
   Using the provided virtual environment (`venv/` or `.venv/`) is also possible.
2. **Run the engine demo**:
   ```bash
   python scripts/smoke_engine.py [--debug]
   ```
   Use `--debug` to print FECR and resonance metrics each turn.
   The loop calls the coordinator at the start and end of each round. Enter text
   to prime the first turn, then press Enter to continue through subsequent rounds
   or type `quit` to exit.

Logs are written to `logs/MRCE_chat_<n>.csv` when `mrce/utils/logger.py` has `CSV_LOGGING = True`.

## Running tests

The project has a small pytest suite checking the Phase‑Crystal memory:

```bash
pytest
```

## License

This project is licensed under the MIT License. See `pyproject.toml` for details.
