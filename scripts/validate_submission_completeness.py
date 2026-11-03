"""Verify that the grading copy preserves every development operation and source file."""
from pathlib import Path
import argparse
import ast
import hashlib
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def verify(path, history_path=None):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    dev = nbformat.read(ROOT/'notebooks/finetuning.ipynb', as_version=4)
    copies = {c.metadata.original_cell_index:c for c in nb.cells
              if 'original_cell_index' in c.metadata}
    expected = {i:c for i,c in enumerate(dev.cells) if c.cell_type=='code'}
    assert set(copies)==set(expected), 'Development operation missing or duplicated'
    for index, original in expected.items():
        cell = copies[index]
        parsed = ast.parse(cell.source)
        assert len(parsed.body)==2
        assignment, guard = parsed.body
        assert isinstance(assignment, ast.Assign) and assignment.value.value is False
        assert isinstance(guard, ast.If) and not guard.orelse
        assert guard.test.id==assignment.targets[0].id
        unwrapped = ast.Module(body=guard.body,type_ignores=[])
        assert ast.dump(unwrapped)==ast.dump(ast.parse(original.source)), index
        assert cell.metadata.original_source_sha256==hashlib.sha256(original.source.encode()).hexdigest()
    sources={c.metadata.source_file:c for c in nb.cells if 'source_file' in c.metadata}
    files={p.relative_to(ROOT).as_posix():p for pattern in ['src/*.py','legacy/*.py','scripts/*.py','tests/*.py'] for p in ROOT.glob(pattern)}
    files['legacy/model_registry.json']=ROOT/'legacy/model_registry.json'
    assert set(sources)==set(files), 'Full implementation source missing'
    for name,p in files.items():
        actual=p.read_text(encoding='utf8')
        assert sources[name].metadata.source_sha256==hashlib.sha256(actual.encode()).hexdigest(),name
        assert actual in sources[name].source,name
    if history_path:
        history=nbformat.read(history_path,as_version=4)
        records={c.metadata.history_cell_index:c for c in nb.cells if 'history_cell_index' in c.metadata}
        assert set(records)==set(range(len(history.cells)))
        for index,c in enumerate(history.cells):
            assert c.source in records[index].source,index
            assert records[index].cell_type=='markdown','Historical approvals must not be executable'
    headings=[c for c in nb.cells if c.cell_type=='markdown' and c.source.startswith('# ')]
    assert len(headings)==4
    print(f'{path.name}: {len(expected)} original executable cells unchanged under False guards; '
          f'{len(files)} complete implementation files; four sections; '
          f'{nb.metadata.submission.read_only_history_cells} read-only history cells')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--private',type=Path)
    parser.add_argument('--history',type=Path)
    args=parser.parse_args()
    verify(ROOT/'notebooks/submission.ipynb')
    if args.private:verify(args.private,args.history)
