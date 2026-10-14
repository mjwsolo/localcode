import urllib.request
from types import SimpleNamespace

import pytest

from localcode.ui.http import bypass_local_proxies, local_urlopen


def test_loopback_request_explicitly_disables_proxies(monkeypatch):
    calls = []
    monkeypatch.setenv('HTTP_PROXY', 'http://proxy.invalid:8080')
    def build(handler):
        assert isinstance(handler, urllib.request.ProxyHandler) and handler.proxies == {}
        return SimpleNamespace(open=lambda request, timeout: calls.append((request, timeout)) or 'response')
    monkeypatch.setattr(urllib.request, 'build_opener', build)
    request = urllib.request.Request('http://127.0.0.1:8323/status', headers={'X-Localcode-Token': 'test'})
    assert local_urlopen(request, timeout=3) == 'response'
    assert calls == [(request, 3)]


@pytest.mark.parametrize('url', ['https://example.com', 'http://127.0.0.1.example.com', 'http://proxy.invalid'])
def test_local_helper_rejects_nonlocal_destinations(url):
    with pytest.raises(ValueError):
        local_urlopen(url)


def test_runtime_bypass_preserves_existing_proxy_exclusions():
    env = {'NO_PROXY': 'example.org', 'no_proxy': 'internal.test,localhost', 'HTTP_PROXY': 'http://proxy.invalid'}
    bypass_local_proxies(env)
    assert env['NO_PROXY'] == env['no_proxy'] == 'example.org,internal.test,localhost,127.0.0.1,::1'
    assert env['HTTP_PROXY'] == 'http://proxy.invalid'
