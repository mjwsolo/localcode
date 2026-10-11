#!/usr/bin/env bash
set -eu
cd "$WORKSPACE"
mkdir -p .eval
cat > account.py <<'PY'
class Account:
    """A named account holding a balance."""

    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError(f"insufficient funds: {self.owner} has {self.balance}, needs {amount}")
        self.balance -= amount
PY
cat > ledger.py <<'PY'
from account import Account


class Ledger:
    """Accounts keyed by owner name, with transfers between them."""

    def __init__(self):
        self._accounts = {}

    def open(self, owner, balance=0):
        account = Account(owner, balance)
        self._accounts[owner] = account
        return account

    def transfer(self, src, dst, amount):
        source, target = self._accounts[src], self._accounts[dst]
        source.withdraw(amount)  # raises ValueError before anything moves
        target.deposit(amount)

    def total(self):
        return sum(a.balance for a in self._accounts.values())
PY
cat > test_ledger.py <<'PY'
import pytest

from account import Account
from ledger import Ledger


def test_deposit_and_withdraw():
    a = Account("x", 10)
    a.deposit(5)
    a.withdraw(7)
    assert a.balance == 8


def test_overdraw_raises():
    with pytest.raises(ValueError):
        Account("x", 10).withdraw(11)


def test_open_returns_account():
    acct = Ledger().open("alice", 100)
    assert isinstance(acct, Account) and acct.balance == 100


def test_transfer_moves_money():
    ledger = Ledger()
    alice, bob = ledger.open("alice", 100), ledger.open("bob")
    ledger.transfer("alice", "bob", 30)
    assert (alice.balance, bob.balance, ledger.total()) == (70, 30, 100)


def test_transfer_insufficient_funds_changes_nothing():
    ledger = Ledger()
    alice, bob = ledger.open("alice", 10), ledger.open("bob")
    with pytest.raises(ValueError):
        ledger.transfer("alice", "bob", 999)
    assert (alice.balance, bob.balance) == (10, 0)
PY
