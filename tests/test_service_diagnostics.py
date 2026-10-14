from localcode import diagnostics, model_files
from localcode.ui import ports


def test_service_report_counts_servers_without_exposing_process_arguments_or_keys(monkeypatch, tmp_path):
    monkeypatch.setattr(model_files, 'model_files', lambda _: {'model.gguf': tmp_path / 'model.gguf'})
    monkeypatch.setattr(diagnostics.subprocess, 'check_output', lambda *a, **k:
                        'PID PPID RSS COMM\n123 10 4096 /private/models/llama-server\n456 10 2000 python\n')
    monkeypatch.setattr(ports, 'find_running', lambda: (8323, {
        'state': 'ready', 'process_id': 123, 'api_key': 'never-print', 'control_token': 'never-print',
    }))
    report = diagnostics.service_report()
    assert report['installed_models'] == 1
    assert report['model_processes'] == [{'pid': 123, 'parent_pid': 10, 'rss_kb': 4096}]
    assert report['server'] == {'state': 'ready', 'process_id': 123}
    assert 'never-print' not in str(report)
