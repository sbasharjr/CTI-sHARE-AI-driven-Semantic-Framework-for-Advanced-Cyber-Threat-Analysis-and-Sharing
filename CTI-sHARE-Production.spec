# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['run_production_server.py'],
    pathex=[],
    binaries=[],
    datas=[('src', 'src'), ('config', 'config'), ('data', 'data'), ('live_threat_data.json', '.')],
    hiddenimports=['waitress', 'flask', 'flask_cors'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='CTI-sHARE-Production',
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
