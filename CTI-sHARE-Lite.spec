# -*- mode: python ; coding: utf-8 -*-
"""
Simplified PyInstaller spec file for CTI-sHARE Production Executable
"""

import sys
import os
from pathlib import Path

# Get the current directory
current_dir = Path(__file__).resolve().parent

# Essential data files only
added_files = [
    # Dashboard templates (essential)
    ('src/dashboard/templates/dashboard.html', 'src/dashboard/templates'),
    
    # Configuration files
    ('config/config.yaml', 'config'),
    
    # Essential data
    ('live_threat_data.json', '.'),
    
    # WSGI configuration
    ('wsgi.py', '.'),
    
    # MISP integration
    ('misp_integration.py', '.'),
]

# Core hidden imports only
hidden_imports = [
    'waitress',
    'flask',
    'flask_cors',
    'requests',
    'json',
    'datetime',
    'logging',
    'threading',
    'queue',
    'time',
    'random',
    'hashlib',
    'urllib.parse',
    'pathlib',
]

# Analysis configuration
a = Analysis(
    ['run_production_server.py'],
    pathex=[str(current_dir)],
    binaries=[],
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy dependencies
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',
        'keras',
        'torch',
        'transformers',
        'nltk',
        'spacy',
        'gensim',
        'matplotlib',
        'plotly',
        'seaborn',
        'jupyter',
        'notebook',
        'ipython',
        'sphinx',
        'pytest',
        'unittest',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
    optimize=1,
)

# Remove duplicate files
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Create executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='CTI-sHARE-Production-Lite',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)