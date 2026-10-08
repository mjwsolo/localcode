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


def test_unsupported_model_never_calls_inference(monkeypatch):
    monkeypatch.setattr("localcode.decision.post", lambda *a: pytest.fail("must not infer"))
    with pytest.raises(DecisionError, match="Select OpenJev"):
        Supervisor.decision(service("Qwen3.8-27B-UD-Q4_K_XL"), build_request("Question", "yes-no", []))


def test_model_switch_during_request_discards_the_result(monkeypatch):
    sup=service()
    def post(*a):
        sup.proc=object()
        return {"answers": {"answer": {"noul": 0.9}}}
    monkeypatch.setattr("localcode.decision.post", post)
    with pytest.raises(DecisionError, match="model changed"):
        Supervisor.decision(sup, build_request("Question", "yes-no", []))
