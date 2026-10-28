# Personal Python Question-Answering Chatbot

A lightweight command-line chatbot that answers Python programming questions in natural language using an indexed knowledge base. Built entirely with the **Python standard library** — zero external dependencies or installs required.

## Features

- **Standard Library Only:** Runs on pure Python (requires Python 3.9+).
- **TF-IDF + Cosine Similarity:** Fast vector search with synonym mapping and spelling correction (`difflib`).
- **Comprehensive Knowledge Base:** 38 indexed Python topics covering basics to advanced internals (GIL, memory management, decorators, async/await, mutable default arguments, `__new__` vs `__init__`, and more).
- **Interactive Chat & CLI Mode:** Chat interactively or pass single-question queries straight from the command line.
- **Dynamic Reload:** Supports a `reload` command inside the chat to update the knowledge base without restarting.

---

## Project Structure

| File | Purpose |
|---|---|
| `chatbot.py` | Chatbot engine: text preprocessing, TF-IDF vectorizer, matching engine, REPL loop |
| `knowledge_base.json` | 38 Python topics with questions, keywords, and verified code examples |
| `run.bat` | One-click launcher for Windows |
| `.gitignore` | Excludes cache, virtual environments, and temporary files |

---

## Getting Started

### 1. Interactive Chat
```bash
python chatbot.py
```
*(On Windows, you can also double-click `run.bat`)*

Commands available inside the chat:
- `topics` — View all 38 topics the bot knows
- `history` — Show questions asked in this session
- `reload` — Refresh knowledge base from disk
- `quit` or `exit` — Exit the chatbot

### 2. Single-Question (One-Shot) Mode
```bash
python chatbot.py "Why shouldn't I use a list as a default argument in a function?"
python chatbot.py "What is the difference between shallow copy and deep copy?"
```

---

## How It Works

1. **Tokenization & Normalization:** Lowercases text, filters stopwords, and applies synonyms (e.g., `dictionaries → dict`, `threading → thread`, `coroutines → async`).
2. **Spelling Correction:** Uses `difflib.get_close_matches` against the vocabulary to auto-correct typos.
3. **TF-IDF Indexing:** Computes Term Frequency-Inverse Document Frequency vectors across topics, questions, keywords, and answers.
4. **Cosine Similarity:** Computes cosine similarity between query and topic vectors to return the best match above confidence thresholds.

---

## Extending the Knowledge Base

You can add new Python topics simply by adding an entry to `knowledge_base.json`:

```json
{
  "topic": "Recursion",
  "questions": ["What is recursion?", "How does a recursive function work?"],
  "keywords": ["recursion", "recursive", "base", "case"],
  "answer": "A recursive function calls itself until reaching a base condition..."
}
```
Then type `reload` in your active chat to immediately load the new topic!
