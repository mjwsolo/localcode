from app.cli import main


def test_greet(capsys):
    assert main(['greet', 'bo']) == 0
    assert capsys.readouterr().out.strip() == 'hello bo'
