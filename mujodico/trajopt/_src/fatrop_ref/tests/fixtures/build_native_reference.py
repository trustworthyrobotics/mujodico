"""Build the optional native FATROP oracle without changing its source tree.

An existing BLASFEO installation is required. The BLASFEO library and headers
bundled with CasADi can be used, without using CasADi's FATROP implementation.
"""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--fatrop-source', type=Path, required=True)
  parser.add_argument(
      '--blasfeo-prefix',
      type=Path,
      help='Directory containing include/blasfeo.h and libblasfeo.so',
  )
  parser.add_argument(
      '--build-dir',
      type=Path,
      default=Path(tempfile.gettempdir()) / 'ippas-native-reference',
  )
  parser.add_argument('--jobs', type=int, default=min(8, os.cpu_count() or 1))
  args = parser.parse_args()
  source = args.fatrop_source.resolve()
  prefix = args.blasfeo_prefix
  if prefix is None:
    try:
      import casadi
    except ImportError as exc:
      parser.error(
          'Pass --blasfeo-prefix, or install CasADi for its BLASFEO dependency'
      )
      raise exc
    prefix = Path(casadi.__file__).parent
  prefix = prefix.resolve()
  library_candidates = [prefix / 'libblasfeo.so', prefix / 'lib/libblasfeo.so']
  library = next((p for p in library_candidates if p.is_file()), None)
  if library is None or not (prefix / 'include/blasfeo.h').is_file():
    parser.error(
        'BLASFEO prefix must contain include/blasfeo.h and libblasfeo.so or'
        ' lib/libblasfeo.so'
    )
  build_dir = args.build_dir.resolve()
  config = build_dir / 'blasfeo-config'
  config.mkdir(parents=True, exist_ok=True)
  (config / 'blasfeoConfig.cmake').write_text(
      'add_library(blasfeo SHARED IMPORTED GLOBAL)\n'
      'set_target_properties(blasfeo PROPERTIES\n'
      f'  IMPORTED_LOCATION "{library}"\n'
      f'  INTERFACE_INCLUDE_DIRECTORIES "{prefix / "include"}")\n'
  )
  native_build = build_dir / 'build'
  subprocess.run(
      [
          'cmake',
          '-S',
          str(source),
          '-B',
          str(native_build),
          '-DBUILD_TESTS=OFF',
          '-DBUILD_EXAMPLES=OFF',
          '-DBUILD_WITH_C_INTERFACE=ON',
          '-DBUILD_SHARED_LIBS=ON',
          '-DCMAKE_BUILD_TYPE=Release',
          f'-Dblasfeo_DIR={config}',
      ],
      check=True,
  )
  subprocess.run(
      ['cmake', '--build', str(native_build), '-j', str(args.jobs)], check=True
  )
  fixture = Path(__file__).with_name('native_reference.cpp')
  executable = build_dir / 'reference'
  subprocess.run(
      [
          os.environ.get('CXX', 'c++'),
          '-O2',
          '-std=c++17',
          str(fixture),
          f'-I{source / "c_interface/include"}',
          f'-I{prefix / "include"}',
          f'-L{native_build / "src"}',
          '-lfatrop',
          f'-L{library.parent}',
          '-lblasfeo',
          f'-Wl,-rpath,{native_build / "src"}',
          f'-Wl,-rpath,{library.parent}',
          '-o',
          str(executable),
      ],
      check=True,
  )
  subprocess.run([str(executable)], check=True)
  print(f'IPPAS_NATIVE_REFERENCE={executable}')


if __name__ == '__main__':
  main()
