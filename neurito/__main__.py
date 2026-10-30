"""Permite abrir a Neurito con ``python -m neurito``.

Es el mismo punto de entrada que ``python -m neurito.pet``; se deja aparte para
que el comando corto funcione sin cambiar el ``main.py`` original del proyecto.
"""

from __future__ import annotations

from .pet import main

if __name__ == "__main__":
    main()
