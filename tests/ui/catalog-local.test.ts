import { expect, test } from "bun:test"
import path from "node:path"

test("catalog groups installed model files without counting projectors or broken links", () => {
  const result = Bun.spawnSync([process.env.LOCALCODE_PY ?? "python3", "-c", `
import pathlib, tempfile
from localcode.ui.supervisor import Supervisor, MODEL_GROUPS
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    group = MODEL_GROUPS[0]
    stem = group.hf_repo.rsplit('/', 1)[-1].removesuffix('-GGUF')
    first = root / (stem + '-Q4_K_M.gguf')
    first.write_bytes(b'test')
    (root / (stem + '-Q8_0.gguf')).write_bytes(b'test')
    (root / ('mmproj-' + stem + '.gguf')).write_bytes(b'test')
    (root / (stem + '-broken.gguf')).symlink_to(root / 'missing')
    (root / (stem + '-directory.gguf')).mkdir()
    supervisor = Supervisor.__new__(Supervisor)
    supervisor.models_dir = root
    supervisor.current = None
    supervisor.ram_gb = 32
    supervisor.state = {'state': 'idle', 'pct': None}
    supervisor.active = {}
    supervisor._rec_repo = lambda: None
    rows = supervisor.catalog()['groups']
    assert next(row for row in rows if row['key'] == group.key)['installed_count'] == 2
    assert sum(row['installed_count'] for row in rows) == 2
    first.unlink()
    assert sum(row['installed_count'] for row in supervisor.catalog()['groups']) == 1
`], { cwd: path.resolve(import.meta.dir, "..") })
  expect(result.stderr.toString()).toBe("")
  expect(result.exitCode).toBe(0)
})
