"""
Personal Python Question-Answering Chatbot
------------------------------------------
Answers Python programming questions from a predefined knowledge base
(knowledge_base.json) using TF-IDF + cosine similarity, implemented with
the standard library only.

Run:  python chatbot.py
"""

import json
import math
import re
import sys
from collections import Counter
from difflib import get_close_matches
from pathlib import Path

KB_PATH = Path(__file__).with_name("knowledge_base.json")
CONFIDENCE_THRESHOLD = 0.18   # below this, the bot says it is unsure
SUGGESTION_THRESHOLD = 0.08   # topics above this are offered as suggestions

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "be", "do", "does", "did", "i", "me",
    "my", "you", "your", "we", "it", "its", "in", "on", "of", "to", "for",
    "and", "or", "how", "what", "why", "when", "which", "can", "could",
    "should", "would", "please", "tell", "explain", "about", "python", "use",
    "using", "way", "some", "there", "this", "that", "between", "get", "make",
}

# Map common alternative words onto the vocabulary used in the knowledge base.
SYNONYMS = {
    "dictionary": "dict", "dictionaries": "dict", "dicts": "dict",
    "lists": "list", "arrays": "list", "array": "list",
    "strings": "string", "str": "string", "text": "string",
    "errors": "error", "exceptions": "exception", "catch": "except",
    "functions": "function", "func": "function", "method": "function",
    "loops": "loop", "looping": "loop", "iterate": "loop", "iterating": "loop",
    "classes": "class", "objects": "object",
    "files": "file", "reading": "read", "writing": "write",
    "tuples": "tuple", "sets": "set", "modules": "module",
    "packages": "package", "libraries": "library",
    "installing": "install", "sorting": "sort", "decorators": "decorator",
    "generators": "generator", "variables": "variable", "types": "type",
    "venv": "virtual", "conversion": "convert", "casting": "convert",
    "cast": "convert", "lambdas": "lambda", "comprehensions": "comprehension",
    "arguments": "argument", "args": "argument", "parameters": "argument",
    "params": "argument", "parameter": "argument", "defaults": "default",
    "copies": "copy", "copying": "copy", "dunder": "magic",
    "threads": "thread", "threading": "thread", "multithreading": "thread",
    "coroutine": "async", "coroutines": "async", "asynchronous": "async",
}

GREETINGS = {"hi", "hello", "hey", "hola", "namaste", "yo"}
THANKS = {"thanks", "thank", "thx", "ty"}


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9_*@=]+", text.lower())
    words = [SYNONYMS.get(w, w) for w in words]
    return [w for w in words if w not in STOPWORDS and len(w) > 1 or w in {"=", "@"}]


class PythonQABot:
    def __init__(self, kb_path: Path = KB_PATH):
        with open(kb_path, encoding="utf-8") as f:
            self.kb = json.load(f)
        self._build_index()
        self.history: list[tuple[str, str]] = []

    # ---------- indexing ----------
    def _build_index(self) -> None:
        docs = []
        for entry in self.kb:
            text = " ".join(entry["questions"]) + " " + entry["topic"]
            # keywords and topic are weighted 3x: they are the strongest signals
            text += (" " + " ".join(entry["keywords"]) + " " + entry["topic"]) * 3
            # answer text provides additional contextual matching
            text += " " + entry.get("answer", "")
            docs.append(Counter(tokenize(text)))

        n = len(docs)
        df = Counter(term for doc in docs for term in doc)
        self.idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
        self.vectors = [self._vectorize(doc) for doc in docs]
        self.vocab = list(self.idf)

    def _vectorize(self, counts: Counter) -> dict[str, float]:
        vec = {t: (1 + math.log(c)) * self.idf.get(t, 0) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    @staticmethod
    def _cosine(a: dict, b: dict) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(t, 0.0) for t, v in a.items())

    # ---------- matching ----------
    def _correct_spelling(self, tokens: list[str]) -> list[str]:
        fixed = []
        for t in tokens:
            if t in self.idf:
                fixed.append(t)
            else:
                match = get_close_matches(t, self.vocab, n=1, cutoff=0.8)
                fixed.append(match[0] if match else t)
        return fixed

    def rank(self, question: str) -> list[tuple[float, dict]]:
        tokens = self._correct_spelling(tokenize(question))
        # "python" is a stopword (it appears in almost every question), so a question
        # that is ONLY about Python itself ("What is Python?") maps to the intro entry.
        if not tokens and "python" in question.lower():
            tokens = ["introduction", "language"]
        if not tokens:
            return []
        qvec = self._vectorize(Counter(tokens))
        scores = [(self._cosine(qvec, v), e) for v, e in zip(self.vectors, self.kb)]
        return sorted(scores, key=lambda s: s[0], reverse=True)

    def answer(self, question: str) -> str:
        words = set(re.findall(r"[a-z]+", question.lower()))
        if words and words <= GREETINGS | {"there", "bot"}:
            return "Hello! Ask me anything about Python. Type 'help' for commands."
        if words & THANKS and len(words) <= 4:
            return "You're welcome! Ask another Python question anytime."

        ranked = self.rank(question)
        if not ranked:
            return "Please type a Python question, e.g. 'How do I reverse a string?'"

        best_score, best = ranked[0]
        if best_score >= CONFIDENCE_THRESHOLD:
            reply = f"[{best['topic']}]\n{best['answer']}"
            related = [e["topic"] for s, e in ranked[1:3] if s >= CONFIDENCE_THRESHOLD]
            if related:
                reply += "\n\nRelated topics: " + ", ".join(related)
            return reply

        suggestions = [e["topic"] for s, e in ranked[:3] if s >= SUGGESTION_THRESHOLD]
        if suggestions:
            return ("I'm not sure I understood. Did you mean one of these topics?\n  - "
                    + "\n  - ".join(suggestions))
        return ("Sorry, I don't have an answer for that yet. Try rephrasing, "
                "or type 'topics' to see what I know.")

    # ---------- chat loop ----------
    def topics(self) -> str:
        return "Topics I can help with:\n  - " + "\n  - ".join(e["topic"] for e in self.kb)

    def run(self) -> None:
        print("=" * 60)
        print("  Python Q&A Chatbot  |  type 'help' for commands")
        print("=" * 60)
        while True:
            try:
                question = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBot: Goodbye!")
                break
            if not question:
                continue
            cmd = question.lower()
            if cmd in {"quit", "exit", "bye", "q"}:
                print("Bot: Goodbye! Happy coding.")
                break
            if cmd == "help":
                reply = ("Commands:\n  topics   - list everything I know\n"
                         "  history  - show this session's questions\n"
                         "  reload   - reload knowledge base from disk\n"
                         "  quit     - exit\nOr just ask a Python question.")
            elif cmd == "reload":
                with open(KB_PATH, encoding="utf-8") as f:
                    self.kb = json.load(f)
                self._build_index()
                reply = "Knowledge base reloaded successfully!"
            elif cmd == "topics":
                reply = self.topics()
            elif cmd == "history":
                reply = ("\n".join(f"{i}. {q}" for i, (q, _) in enumerate(self.history, 1))
                         or "No questions yet.")
            else:
                reply = self.answer(question)
                self.history.append((question, reply))
            print(f"Bot: {reply}")


if __name__ == "__main__":
    bot = PythonQABot()
    if len(sys.argv) > 1:                    # one-shot mode: python chatbot.py "question"
        print(bot.answer(" ".join(sys.argv[1:])))
    else:
        bot.run()
