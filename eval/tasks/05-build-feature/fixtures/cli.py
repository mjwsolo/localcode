import typer

app = typer.Typer()


@app.command()
def greet(name: str = "world"):
    """Print a greeting."""
    typer.echo(f"hello, {name}")


if __name__ == "__main__":
    app()
