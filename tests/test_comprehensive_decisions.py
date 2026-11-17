"""Decision-mode boundaries: typed inference only, model checks, input and outputs."""
import types

import pytest

from localcode.decision import DecisionError, build_request, supported, validate_request, validate_response
from localcode.ui.supervisor import Supervisor


def test_only_decision_models_are_enabled():
    assert supported("OpenJev-Q4_K_M")
    assert supported("localcode/OpenJev-Q8_0")
    assert not supported("Qwen3.8-27B-UD-Q4_K_XL")


@pytest.mark.parametrize("kind,options", [("choice", ["billing"]), ("choice", ["billing", "billing"]), ("score", ["low"]), ("yes-no", ["yes"])] )
def test_invalid_options_are_refused(kind, options):
    with pytest.raises(DecisionError):
        build_request("Decide", kind, options)


def test_requests_use_the_typed_protocol_not_chat_messages():
    request = build_request("Which team?", "choice", ["billing", "shipping"], "Charged twice")
    assert "messages" not in request
    assert request["questions"]["answer"] == {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "shipping": None}}


@pytest.mark.parametrize("mutation", ["remote-image", "unknown-type", "missing-question", "too-many-levels"])
def test_bad_requests_fail_before_inference(mutation):
    request = build_request("Is it urgent?", "yes-no", [])
    if mutation == "remote-image": request["images"] = ["https://example.invalid/private.png"]
    if mutation == "unknown-type": request["questions"]["answer"]["type"] = "chat"
    if mutation == "missing-question": request["questions"]["answer"]["instructions"] = ""
    if mutation == "too-many-levels": request["questions"]["answer"] = {"type": "score", "instructions": "Score", "criteria": ["level"] * 11}
    with pytest.raises(DecisionError): validate_request(request)


def test_chat_responses_and_invalid_probabilities_are_not_accepted():
    request = build_request("Is it urgent?", "yes-no", [])
    for response in [{"choices": [{"message": {"content": "Yes"}}]}, {"answers": {"answer": {"noul": float("nan")}}}, {"answers": {"answer": {"noul": True}}}]:
        with pytest.raises(DecisionError): validate_response(response, request)


def service(alias="OpenJev-Q4_K_M"):
    return types.SimpleNamespace(state={"state": "ready"}, current=alias, proc=object(), port=8123)


def test_supervisor_uses_only_systemone(monkeypatch):
    calls = []
    def post(url, body, headers):
        calls.append((url,body))
        return {"answers": {"answer": {"noul": 0.9}}}
    monkeypatch.setattr("localcode.decision.post", post)
    result = Supervisor.decision(service(), build_request("Is it urgent?", "yes-no", []))
    assert result["answers"]["answer"]["noul"] == 0.9
    assert calls[0][0] == "http://127.0.0.1:8123/v1/systemone"


def _swap_service(alias="Qwen3.8-27B-UD-Q4_K_XL", downloaded=True):
    sup = service(alias)
    sup.active = {"group": "qwen38", "filename": alias + ".gguf"}
    sup.selected = []
    sup.decision_model = lambda: {"group": "openjev", "filename": "OpenJev-Q4_K_M.gguf", "name": "OpenJev 27B",
                                  "size_gb": 19.0, "downloaded": downloaded}
    def select(group, filename):
        sup.selected.append((group, filename))
        sup.current, sup.proc, sup.state = filename.removesuffix(".gguf"), object(), {"state": "ready"}
        return {"ok": True}
    sup.select = select
    sup.decision_setup = lambda: Supervisor.decision_setup(sup)
    sup.decision_restore = lambda: Supervisor.decision_restore(sup)
    sup._await_ready = lambda alias: Supervisor._await_ready(sup, alias, timeout=1)
    return sup


def test_missing_decision_model_is_never_downloaded_by_a_decision(monkeypatch):
    monkeypatch.setattr("localcode.decision.post", lambda *a: pytest.fail("must not infer"))
    sup = _swap_service(downloaded=False)
    with pytest.raises(DecisionError, match=r"not downloaded yet \(19.0 GB\).*/decide"):
        Supervisor.decision(sup, build_request("Question", "yes-no", []))
    assert sup.selected == []


def test_downloaded_decision_model_is_swapped_in_and_the_chat_model_restored(monkeypatch):
    monkeypatch.setattr("localcode.decision.post", lambda *a: {"answers": {"answer": {"noul": 0.7}}})
    sup = _swap_service()
    result = Supervisor.decision(sup, build_request("Question", "yes-no", []))
    assert result["model"] == "OpenJev-Q4_K_M"
    assert sup.selected == [("openjev", "OpenJev-Q4_K_M.gguf")]
    assert Supervisor.decision_restore(sup)["ok"]
    assert sup.selected[-1] == ("qwen38", "Qwen3.8-27B-UD-Q4_K_XL.gguf")
    # Nothing left to restore once the chat model is back.
    assert Supervisor.decision_restore(sup) == {"ok": True, "restored": False}


def test_picker_catalog_hides_decision_only_models():
    from localcode.models_catalog import MODEL_GROUPS
    sup = Supervisor.__new__(Supervisor)
    sup.current, sup.ram_gb, sup.active = None, 64, {}
    sup.state = {"state": "idle", "pct": None}
    sup._installed_models = lambda: {}
    sup._rec_repo = lambda: None
    sup._current_repo = lambda: None
    sup.models_dir = __import__("pathlib").Path("/nonexistent")
    keys = {g["key"] for g in sup.catalog()["groups"]}
    assert "openjev" not in keys
    assert keys == {g.key for g in MODEL_GROUPS if not g.supports_systemone}


def test_decision_model_reports_its_size_and_download_state(tmp_path):
    sup = Supervisor.__new__(Supervisor)
    sup.models_dir = tmp_path
    info = Supervisor.decision_model(sup)
    assert info["group"] == "openjev" and info["filename"] == "OpenJev-Q4_K_M.gguf"
    assert info["size_gb"] > 10 and info["name"].startswith("OpenJev")


def test_model_switch_during_request_discards_the_result(monkeypatch):
    sup=service()
    def post(*a):
        sup.proc=object()
        return {"answers": {"answer": {"noul": 0.9}}}
    monkeypatch.setattr("localcode.decision.post", post)
    with pytest.raises(DecisionError, match="model changed"):
        Supervisor.decision(sup, build_request("Question", "yes-no", []))
