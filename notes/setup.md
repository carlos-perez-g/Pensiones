# Setup del entorno

## Para replicar en tu Mac (terminal)

```bash
cd "ruta/a/Claude-Cowork"
pip install torch matplotlib numba ipython pynvml EconModel consav
pip install -e EconDLSolvers/.
python3 model/test_setup_simpleconsav.py   # debe terminar con "TEST OK"
```

En Mac, `pip install torch` instala la versión CPU/MPS (no hay CUDA en Apple Silicon).

## Estado del entorno en el sandbox de Claude (2026-07-10)

- Python 3.10.12, torch 2.8.0+cpu (wheel CPU-only aarch64; el wheel por defecto
  de PyPI para esta arquitectura exige librerías CUDA y no importa sin GPU).
- consav, EconModel, numba, ipython, pynvml, matplotlib instalados.
- EconDLSolvers instalado editable desde el clon local.
- Nota: el sandbox se reinicia entre sesiones; el entorno hay que reinstalarlo
  cada sesión (los archivos del proyecto persisten, los paquetes no).

## Test de humo

`model/test_setup_simpleconsav.py`: modelo consumo-ahorro simple (T=3, utilidad
log) del repo, resuelto con DeepSimulate, 0.3 min de entrenamiento en CPU.
Verificación: la política terminal cumple c_T = m_T exactamente (es impuesta
analíticamente por `terminal_actions`, así que valida el pipeline, no la
convergencia). R alcanzado ≈ -2.184.

## Entrenamientos largos

- Corridas cortas de desarrollo: en el sandbox de Claude o tu Mac.
- Corridas largas / modelo final: tu Mac (CPU/MPS) o GPU arrendada en
  runpod.io (ver `EconDLSolvers/runpod.md`).
