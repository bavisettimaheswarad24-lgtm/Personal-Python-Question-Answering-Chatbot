# Personal Python Question-Answering Chatbot

A command-line chatbot that answers Python programming questions in natural language, using a predefined knowledge base. Pure Python standard library — no installs needed.

## Files

| File | Purpose |
|---|---|
| `chatbot.py` | The chatbot: text processing, matching engine, chat loop |
| `knowledge_base.json` | 31 Python topics, each with sample questions, keywords and an answer |

## Run it

```bash
python chatbot.py                               # interactive chat
python chatbot.py "how do I reverse a string"   # one-shot answer
```

Requires Python 3.9+.

Commands inside the chat: `help`, `topics`, `history`, `quit`.

## How it works

1. **Preprocessing** — the question is lower-cased, tokenized, stopwords are removed, and synonyms are normalized (e.g. `dictionary → dict`, `errors → error`, `iterate → loop`).
2. **Spelling correction** — unknown words are matched to the closest known word with `difflib` (e.g. `diffrence → difference`).
3. **TF-IDF vectors** — every knowledge-base entry (its sample questions, topic and keywords, with keywords weighted higher) is turned into a TF-IDF vector when the bot starts.
4. **Cosine similarity** — the question vector is compared with every entry; the highest score wins.
5. **Confidence handling**
   - score ≥ 0.20 → answer, plus up to two related topics
   - 0.08–0.20 → "Did you mean…" suggestions
   - below that → polite fallback
6. **Small talk** — greetings and thanks get friendly replies.

## Example session

```
You: diffrence between list and tuple
Bot: [Tuples]
A tuple is an ordered, immutable collection ...

You: how to catch errors
Bot: [Exception handling]
Use try/except to catch errors ...

You: how to make pizza
Bot: Sorry, I don't have an answer for that yet ...
```

## Extending the knowledge base

Add an object to `knowledge_base.json` — no code changes needed:

```json
{
  "topic": "Recursion",
  "questions": ["What is recursion?", "How does a recursive function work?"],
  "keywords": ["recursion", "recursive", "base", "case"],
  "answer": "A recursive function calls itself ..."
}
```

## Possible improvements

- Swap TF-IDF for sentence embeddings (e.g. `sentence-transformers`) for better understanding of paraphrases
- Add a web UI with Flask or Streamlit
- Log unanswered questions to grow the knowledge base
