#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
cat > calc.py <<'PY'
"""Arithmetic expression evaluator: a tokenizer and a recursive-descent parser."""

DIGITS = "0123456789"


def _tokenize(expr):
    if not isinstance(expr, str):
        raise ValueError("expression must be a string")
    tokens, i, n = [], 0, len(expr)
    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
        elif ch in "+-*/()":
            tokens.append(ch)
            i += 1
        elif ch in DIGITS or ch == ".":
            j = i
            while j < n and (expr[j] in DIGITS or expr[j] == "."):
                j += 1
            text = expr[i:j]
            if text.count(".") > 1 or text == ".":
                raise ValueError(f"bad number {text!r}")
            tokens.append(float(text) if "." in text else int(text))
            i = j
        else:
            raise ValueError(f"unexpected character {ch!r}")
    return tokens


class _Parser:
    def __init__(self, tokens):
        self.tokens, self.pos = tokens, 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def expr(self):
        value = self.term()
        while self.peek() in ("+", "-"):
            if self.take() == "+":
                value = value + self.term()
            else:
                value = value - self.term()
        return value

    def term(self):
        value = self.unary()
        while self.peek() in ("*", "/"):
            if self.take() == "*":
                value = value * self.unary()
            else:
                value = value / self.unary()
        return value

    def unary(self):
        if self.peek() == "-":
            self.take()
            return -self.unary()
        return self.primary()

    def primary(self):
        tok = self.take()
        if isinstance(tok, (int, float)):
            return tok
        if tok == "(":
            value = self.expr()
            if self.take() != ")":
                raise ValueError("missing closing parenthesis")
            return value
        raise ValueError("unexpected end of input" if tok is None else f"unexpected token {tok!r}")


def evaluate(expr):
    """Evaluate an arithmetic expression string; raise ValueError if it is malformed."""
    parser = _Parser(_tokenize(expr))
    if not parser.tokens:
        raise ValueError("empty expression")
    value = parser.expr()
    if parser.peek() is not None:
        raise ValueError(f"unexpected token {parser.peek()!r}")
    return value
PY
