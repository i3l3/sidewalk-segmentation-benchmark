from pathlib import Path
import ast
import nbformat

root=Path(__file__).resolve().parents[1]
for path in sorted((root/'notebooks').glob('*.ipynb')):
    nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    headings=[cell.source.splitlines()[0] for cell in nb.cells if cell.cell_type=='markdown' and cell.source.startswith('# ')]
    assert len(headings)==4,(path,headings)
    for cell in nb.cells:
        if cell.cell_type=='code':
            ast.parse(cell.source)
            assert 'PACKAGE_BASE64' not in cell.source and 'b64decode' not in cell.source
    assert 'git' in nb.cells[2].source and 'REPO_URL' in nb.cells[2].source
    print(path.name,': four sections, valid Python, readable Git clone bootstrap')
