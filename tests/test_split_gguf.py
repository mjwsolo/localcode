"""Split (sharded) GGUF quants: listing, download, installed detection, load path."""
import threading
from pathlib import Path

import pytest

from localcode import bootstrap, hf_quants, models_catalog, split_gguf
from localcode.model_files import model_files
from localcode.ui import supervisor

GB = 10 ** 9


def entry(path, size):
    return {"path": path, "size": size, "lfs": {"size": size}}


def flint_tree():
    return [
        entry("README.md", 1000),
        entry("flint-Q4_K_M.gguf", 4 * GB),
        entry("mmproj-F16.gguf", GB),
        # split in a subfolder
        entry("BF16/flint-BF16-00001-of-00003.gguf", 1_000_000),
        entry("BF16/flint-BF16-00002-of-00003.gguf", 49_900_000_000),
        entry("BF16/flint-BF16-00003-of-00003.gguf", 29_600_000_000),
        # split at the top level
        entry("flint-UD-Q8_K_XL-00001-of-00002.gguf", 30 * GB),
        entry("flint-UD-Q8_K_XL-00002-of-00002.gguf", 5 * GB),
        # sidecar splits and unrelated subfolders stay out
        entry("MTP/flint-MTP-Q4_0-00001-of-00002.gguf", GB),
        entry("MTP/flint-MTP-Q4_0-00002-of-00002.gguf", GB),
        entry("mmproj-BF16-00001-of-00002.gguf", GB),
        entry("mmproj-BF16-00002-of-00002.gguf", GB),
        entry("old/flint-Q2_K.gguf", GB),
        # a split with a missing shard is not listed
        entry("Q6/flint-Q6_K-00001-of-00003.gguf", GB),
        entry("Q6/flint-Q6_K-00003-of-00003.gguf", GB),
    ]


def test_listing_groups_shards_into_one_quant():
    entries = flint_tree()
    quants = [q for e in entries if (q := hf_quants._quant_from_entry(e))] + hf_quants._split_quants(entries)
    by_label = {q.label: q for q in quants}
    assert sorted(by_label) == ["BF16", "Q4_K_M", "UD-Q8_K_XL"]
    bf16 = by_label["BF16"]
    assert bf16.filename == "flint-BF16-00001-of-00003.gguf"
    assert bf16.size_gb == pytest.approx(79.501, abs=0.001)
    assert [p for p, _ in bf16.shards] == [
        "BF16/flint-BF16-00001-of-00003.gguf",
        "BF16/flint-BF16-00002-of-00003.gguf",
        "BF16/flint-BF16-00003-of-00003.gguf",
    ]
    top = by_label["UD-Q8_K_XL"]
    assert top.size_gb == 35.0 and len(top.shards) == 2
    assert by_label["Q4_K_M"].shards == ()


def test_fetch_quants_lists_split_and_caches_shards(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALCODE_HOME", str(tmp_path))

    class Resp:
        def raise_for_status(self): pass
        def json(self): return flint_tree()

    monkeypatch.setattr(hf_quants.httpx, "get", lambda *a, **k: Resp())
    quants = hf_quants.fetch_quants("unsloth/flint-GGUF")
    assert [q.label for q in quants] == ["Q4_K_M", "UD-Q8_K_XL", "BF16"]
    cached, _ = hf_quants._read_cache("unsloth/flint-GGUF")
    assert cached == quants
    assert isinstance(cached[2].shards[0], tuple)


def test_label_and_alias_of_a_shard():
    assert hf_quants._parse_label("BF16/flint-BF16-00001-of-00003.gguf") == "BF16"
    assert split_gguf.display_alias("flint-BF16-00001-of-00003.gguf") == "flint-BF16"
    assert split_gguf.display_alias("flint-Q4_K_M.gguf") == "flint-Q4_K_M"
    assert models_catalog._quant_label("flint-UD-Q8_K_XL-00001-of-00002.gguf") == "UD-Q8_K_XL"


def make_split(directory: Path, stem="flint-BF16", sizes=(10, 20, 30)):
    n = len(sizes)
    names = [f"{stem}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]
    for name, size in zip(names, sizes):
        (directory / name).write_bytes(b"x" * size)
    return names


def test_split_complete_needs_every_shard_at_its_size(tmp_path):
    names = make_split(tmp_path)
    first = tmp_path / names[0]
    sizes = {n: s for n, s in zip(names, (10, 20, 30))}
    assert split_gguf.split_complete(first, sizes)
    assert split_gguf.total_bytes(first) == 60
    (tmp_path / names[2]).write_bytes(b"x" * 29)  # short shard
    assert not split_gguf.split_complete(first, sizes)
    (tmp_path / names[2]).write_bytes(b"x" * 30)
    (tmp_path / names[1]).unlink()  # missing shard
    assert not split_gguf.split_complete(first, sizes)


def test_model_files_lists_only_complete_splits_by_first_shard(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    d = tmp_path / "models"
    d.mkdir()
    names = make_split(d)
    (d / "plain-Q4_K_M.gguf").write_bytes(b"x" * 5)
    found = model_files(d)
    assert names[0] in found and names[1] not in found and names[2] not in found
    (d / names[1]).unlink()
    assert names[0] not in model_files(d)


def test_load_path_is_the_first_shard(tmp_path):
    names = make_split(tmp_path)
    assert split_gguf.resolve_model_path(tmp_path, "flint-BF16") == tmp_path / names[0]
    (tmp_path / "flint-Q4_K_M.gguf").write_bytes(b"x")
    assert split_gguf.resolve_model_path(tmp_path, "flint-Q4_K_M") == tmp_path / "flint-Q4_K_M.gguf"
    sup = supervisor.Supervisor.__new__(supervisor.Supervisor)
    sup.models_dir = tmp_path
    assert sup._model_path("flint-BF16").name == "flint-BF16-00001-of-00003.gguf"


def test_link_and_remove_cover_all_shards(tmp_path):
    src, dst = tmp_path / "hub", tmp_path / "models"
    src.mkdir()
    names = make_split(src)
    split_gguf.link_split(src / names[0], dst)
    assert sorted(p.name for p in dst.iterdir()) == names
    assert split_gguf.remove_split(dst / names[0]) == names
    assert not list(dst.iterdir())
    assert all((src / n).exists() for n in names)


def group_and_stem():
    g = supervisor.MODEL_GROUPS[0]
    stem = g.hf_repo.rsplit("/", 1)[-1].removesuffix("-GGUF")
    return g, stem


def split_quant(stem, sizes=(10, 20, 30), sub="BF16"):
    n = len(sizes)
    shards = tuple((f"{sub}/{stem}-BF16-{i:05d}-of-{n:05d}.gguf", s) for i, s in enumerate(sizes, 1))
    return hf_quants.Quant(shards[0][0].rsplit("/", 1)[-1], sum(sizes) / 1e9, "BF16", False, shards)


def test_picker_row_downloaded_only_when_all_shards_present(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    g, stem = group_and_stem()
    q = split_quant(stem)
    sup = supervisor.Supervisor.__new__(supervisor.Supervisor)
    sup.models_dir = tmp_path; sup.current = None
    sup.ram_gb = 64; sup.bandwidth = 150
    sup.state = {"state": "idle", "pct": None}; sup.active = {}
    sup._quant_rows = lambda group, refresh=False: [q]

    def row():
        out = sup.quants(g.key)["quants"]
        return next(r for r in out if r["label"] == "BF16")

    assert row()["downloaded"] is False
    assert row()["alias"] == f"{stem}-BF16"
    names = [p.rsplit("/", 1)[-1] for p, _ in q.shards]
    for n, (_, size) in zip(names[:2], q.shards[:2]):
        (tmp_path / n).write_bytes(b"x" * size)
    assert row()["downloaded"] is False  # last shard missing
    (tmp_path / names[2]).write_bytes(b"x" * 29)
    sup._model_file_cache = None
    assert row()["downloaded"] is False  # short shard
    (tmp_path / names[2]).write_bytes(b"x" * 30)
    sup._model_file_cache = None
    assert row()["downloaded"] is True
    assert sup._model_ready(q.filename, q)


def fake_download(content_by_url, calls):
    def download(url, dest, num_threads=16, on_progress=None, cancel_event=None):
        calls.append(url)
        data = content_by_url[url.split("/resolve/main/", 1)[1]]
        half = len(data) // 2
        dest.write_bytes(data[:half])
        if on_progress:
            on_progress(f"Downloading: {half // 1048576}/{len(data) // 1048576} MB (50%)")
        dest.write_bytes(data)
    return download


def choice_for(g, q):
    return models_catalog.choice_for_quant(g, q.filename, q.size_gb, q.shards)


def test_download_fetches_every_shard_flat_with_combined_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALCODE_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(bootstrap, "get_model_path", lambda *_: None)
    g, stem = group_and_stem()
    q = split_quant(stem, sizes=(2 * 1048576, 6 * 1048576, 4 * 1048576))
    data = {p: b"y" * s for p, s in q.shards}
    calls, msgs = [], []
    monkeypatch.setattr(bootstrap, "_download_parallel", fake_download(data, calls))
    monkeypatch.setattr(bootstrap, "_verify_download_integrity", lambda *_: (True, ""))
    ok, res = bootstrap.download_model(choice_for(g, q), on_progress=msgs.append)
    assert ok
    models = tmp_path / "models"
    assert Path(res) == models / q.filename
    assert sorted(p.name for p in models.iterdir()) == sorted(p.rsplit("/", 1)[-1] for p, _ in q.shards)
    assert [c.split("/resolve/main/", 1)[1] for c in calls] == [p for p, _ in q.shards]
    progress = [m for m in msgs if m.startswith("Downloading:")]
    assert any("/12 MB" in m for m in progress)  # total is the sum of shards
    # an already-complete shard is skipped on a second run
    calls.clear()
    (models / q.filename).unlink()
    ok, _ = bootstrap.download_model(choice_for(g, q), on_progress=msgs.append)
    assert ok and len(calls) == 1


def test_download_cancel_stops_and_keeps_finished_shards(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALCODE_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(bootstrap, "get_model_path", lambda *_: None)
    g, stem = group_and_stem()
    q = split_quant(stem, sizes=(1048576, 1048576))
    cancel = threading.Event()
    calls = []

    def download(url, dest, **kw):
        calls.append(url)
        dest.write_bytes(b"z" * 1048576)
        cancel.set()

    monkeypatch.setattr(bootstrap, "_download_parallel", download)
    ok, res = bootstrap.download_model(choice_for(g, q), cancel_event=cancel)
    assert (ok, res) == (False, "cancelled")
    assert len(calls) == 1


def test_short_shard_after_download_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALCODE_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(bootstrap, "get_model_path", lambda *_: None)
    g, stem = group_and_stem()
    q = split_quant(stem, sizes=(1048576, 1048576))
    monkeypatch.setattr(bootstrap, "_download_parallel", lambda url, dest, **kw: dest.write_bytes(b"s"))
    ok, res = bootstrap.download_model(choice_for(g, q))
    assert not ok and "incomplete" in res
