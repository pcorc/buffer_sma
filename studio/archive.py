"""Persist a completed, self-contained report on the machine running Streamlit."""
from datetime import datetime
from io import BytesIO
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
from uuid import uuid4
from zipfile import ZipFile

DEFAULT_ARCHIVE_ROOT = r'G:\Shared drives\Portfolio Management\Funds\Archive\Model Portfolio Allocation\ECR Backtest Results'


def archive_root_path(root):
    """Validate a destination on the machine running the app or research script."""
    root_text = str(root).strip()
    if not root_text:
        raise OSError('Choose an archive folder.')
    if os.name != 'nt' and PureWindowsPath(root_text).drive:
        raise OSError('This Windows shared drive is not available on this computer. Choose an accessible --archive-root.')
    base = Path(root_text).expanduser()
    if not base.is_absolute():
        raise OSError('The archive location must be an absolute path.')
    if os.name == 'nt' and not Path(base.anchor).exists():
        raise OSError(f'The drive {base.anchor} is not connected.')
    return base


def new_archive_path(root, run_name, strategy_id):
    base = archive_root_path(root)
    name = re.sub(r'[^A-Za-z0-9_-]+', '-', run_name).strip('-')[:45] or 'ECR'
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    return base / f'{stamp}_{name}_{strategy_id[:8]}_{uuid4().hex[:6]}'


def save_archive(downloads, root, run_name, strategy_id, *, folder_name=None):
    """Create a unique folder; publish success only after every file is written.

    A failed save raises OSError; callers retain the in-memory ZIP for download.
    A Windows drive path must never become a literal relative folder on Linux.
    """
    if folder_name is not None:
        if not re.fullmatch(r'[0-9]{3,6}', folder_name):
            raise ValueError('Batch folder name must be a short numeric identifier.')
        folder = archive_root_path(root) / folder_name
    else:
        folder = new_archive_path(root, run_name, strategy_id)
    created = False
    try:
        folder.mkdir(parents=True, exist_ok=False)
        created = True
        with ZipFile(BytesIO(downloads['zip'])) as archive:
            for entry in archive.infolist():
                target = folder / entry.filename
                if not target.resolve().is_relative_to(folder.resolve()):
                    raise OSError('Invalid report archive path.')
            archive.extractall(folder)
        (folder / 'results.zip').write_bytes(downloads['zip'])
        (folder / 'ARCHIVE_COMPLETE.txt').write_text('Completed report archive. See manifest.json for the run configuration and source hashes.\n')
        return folder
    except Exception:
        if created:
            shutil.rmtree(folder, ignore_errors=True)
        raise
