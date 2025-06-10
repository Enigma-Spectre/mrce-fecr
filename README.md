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
   pip install numpy httpx pyyaml rich
   ```
   Using the provided virtual environment (`venv/` or `.venv/`) is also possible.
2. **Run the engine demo**:
   ```bash
   python scripts/smoke_engine.py
   ```
   Type messages at the prompt and the coordinator will route them through the agents.

Logs are written to `logs/MRCE_chat_<n>.csv` when `mrce/utils/logger.py` has `CSV_LOGGING = True`.

## Running tests

The project has a small pytest suite checking the Phase‑Crystal memory:

```bash
pytest
```

## License

This project is licensed under the MIT License. See `pyproject.toml` for details.
