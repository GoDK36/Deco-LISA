# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(['UI_LISA.py'],
             pathex=['D:\\Shin\\Deco-Lab\\Programs\\Deco-NERO-New'],
             binaries=[],
             datas=[('./UI/LISA.ui','.'),('./UI/LISA_res.ui','.'),('./LINITO.png','.'),('./Htmls','./Htmls'),('./Freq','./Freq'),('./NERs.ico','.'),('./UI/DICORA-logo.png','.'),('./UI/DICORA_INFO.ui','.')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='Deco-LISA',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True, icon='NERs.ico' )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='Deco-LISA')
