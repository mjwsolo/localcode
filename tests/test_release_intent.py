"""Version-bump release triggers, exercising real before/after commits."""
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/release_intent.py'


@pytest.fixture
def repo(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'pyproject.toml').write_text('[project]\nversion="0.5.24"\n')
    subprocess.run(['git','add','.'], cwd=tmp_path, check=True)
    subprocess.run(['git','-c','user.name=Release tests','-c','user.email=tests@example.invalid',
                    'commit','-qm','Before'], cwd=tmp_path, check=True)
    before = subprocess.check_output(['git','rev-parse','HEAD'], cwd=tmp_path,text=True).strip()
    return tmp_path, before


def intent(repo, event='push', ref='refs/heads/main'):
    directory, before = repo
    return subprocess.check_output([os.sys.executable,str(SCRIPT),'--event',event,'--ref',ref,'--before',before],
                                   cwd=directory,text=True).strip()


def test_main_publishes_only_version_changes(repo):
    assert intent(repo) == 'publish=false'
    (repo[0] / 'pyproject.toml').write_text('[project]\nversion="0.5.25"\n')
    assert intent(repo) == 'publish=true'
    assert intent(repo, ref='refs/heads/feature') == 'publish=false'


@pytest.mark.parametrize('event,ref', [('workflow_dispatch','refs/heads/main'),('push','refs/tags/v0.5.25'),('release','refs/tags/v0.5.25')])
def test_explicit_release_paths_still_publish(repo,event,ref):
    assert intent(repo,event,ref) == 'publish=true'


def test_publish_workflow_has_automatic_main_version_gate():
    text = (SCRIPT.parent.parent / '.github/workflows/publish.yml').read_text()
    assert 'branches: [main]' in text
    assert 'needs: [intent]' in text
    assert "if: needs.intent.outputs.publish == 'true'" in text
