import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from src.aihub_download import parse_catalog, selection, check_space, download_files

CATALOG = """└─ Training
    ├─ 원천데이터
    │   └─TS1.zip | 1.5 GB | 101
    └─ 라벨링데이터
        └─TL1.zip | 256 MB | 102
"""

class DownloadTests(unittest.TestCase):
    def test_catalog(self):
        rows = parse_catalog(CATALOG)
        self.assertEqual([r['file_key'] for r in rows], ['101','102'])
        self.assertEqual(rows[0]['size_bytes'], int(1.5*1024**3))
        self.assertIn('원천데이터',rows[0]['path'])
        self.assertIn('라벨링데이터',rows[1]['path'])
        self.assertNotIn('원천데이터',rows[1]['path'])

    def test_selection_guards(self):
        for keys in [[], ['all'], ['999'], ['101','101']]:
            with self.assertRaises(ValueError):selection(parse_catalog(CATALOG),keys)

    def test_disk_guard(self):
        with patch('src.aihub_download.shutil.disk_usage',return_value=Mock(free=1)):
            with self.assertRaises(ValueError):check_space('.',parse_catalog(CATALOG))

    def test_zero_exit_http_failure_redacts_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake=Mock(stdout=io.StringIO('Download failed with HTTP status 403. secret-value\n'))
            fake.wait.return_value=0
            with patch('src.aihub_download.list_files',return_value=parse_catalog(CATALOG)), patch('src.aihub_download.install_shell',return_value=Path('/tmp/tool')), patch('src.aihub_download.check_space'), patch('src.aihub_download.subprocess.Popen',return_value=fake) as popen:
                with self.assertRaises(RuntimeError):download_files(tmp,tmp,['101'],'secret-value')
                command=popen.call_args.args[0]
                self.assertNotIn('secret-value',str(command))
            self.assertFalse(list(Path(tmp).rglob('download_manifest.json')))
            self.assertNotIn('secret-value',next(Path(tmp).rglob('download.log')).read_text())

    def test_success_records_archives_without_secret(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            def launch(*args, **kwargs):
                with zipfile.ZipFile(Path(kwargs['cwd'])/'TS1.zip','w') as archive:
                    archive.writestr('image.txt','fixture')
                fake=Mock(stdout=io.StringIO('Request successful with HTTP status 200.\n'));fake.wait.return_value=0
                return fake
            with patch('src.aihub_download.list_files',return_value=parse_catalog(CATALOG)), patch('src.aihub_download.install_shell',return_value=Path('/tmp/tool')), patch('src.aihub_download.check_space'), patch('src.aihub_download.subprocess.Popen',side_effect=launch):
                result=download_files(tmp,tmp,['101'],'secret-value')
            self.assertEqual(result['status'],'downloaded_archives')
            manifest=next(Path(tmp).rglob('download_manifest.json')).read_text(encoding='utf-8')
            self.assertNotIn('secret-value',manifest)
            with self.assertRaises(FileExistsError), patch('src.aihub_download.list_files',return_value=parse_catalog(CATALOG)), patch('src.aihub_download.check_space'):
                download_files(tmp,tmp,['101'],'secret-value')

    def test_missing_archive_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake=Mock(stdout=io.StringIO('Request successful with HTTP status 200.\n'));fake.wait.return_value=0
            with patch('src.aihub_download.list_files',return_value=parse_catalog(CATALOG)), patch('src.aihub_download.install_shell',return_value=Path('/tmp/tool')), patch('src.aihub_download.check_space'), patch('src.aihub_download.subprocess.Popen',return_value=fake):
                with self.assertRaises(RuntimeError):download_files(tmp,tmp,['101'],'secret')
            self.assertFalse(list(Path(tmp).rglob('download_manifest.json')))

if __name__=='__main__':unittest.main()
