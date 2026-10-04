"""Check dependency compatibility, including NVIDIA's legacy ARM64 SBSA wheel tag."""
import importlib.metadata as metadata
from pathlib import Path
import platform
import subprocess
import sys

result = subprocess.run(['uv','pip','check','--python',sys.executable], text=True, capture_output=True)
output = result.stdout + result.stderr
print(output, end='')
if result.returncode:
    warning = 'The package `nvidia-cusparselt-cu13` was built for a different platform'
    lines = output.strip().splitlines()
    if platform.machine() != 'aarch64' or lines[-2:] != ['Found 1 incompatibility',warning]:
        raise SystemExit(result.returncode)
    package = metadata.distribution('nvidia-cusparselt-cu13')
    if package.version != '0.8.1' or 'Tag: py3-none-manylinux2014_sbsa' not in package.read_text('WHEEL'):
        raise SystemExit(result.returncode)
    libraries = [Path(package.locate_file(p)) for p in package.files if str(p).endswith('libcusparseLt.so.0')]
    if len(libraries) != 1:
        raise SystemExit(result.returncode)
    with libraries[0].open('rb') as stream:
        header = stream.read(20)
    if header[:6] != b'\x7fELF\x02\x01' or int.from_bytes(header[18:20],'little') != 183:
        raise SystemExit('NVIDIA SBSA library is not an AArch64 ELF binary')
    print('Accepted NVIDIA0.8.1 SBSA metadata tag after verifying its AArch64 binary; no other dependency errors.')
