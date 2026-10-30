"""Prepara las ilustraciones para el repositorio.

Los PNG originales pesan ~1.2 MB cada uno. Este script recorta el espacio
transparente, centra al personaje, reduce a 512 px y guarda una copia optimizada
en ``assets/pet/``. Los originales no se modifican.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "assets"
DESTINATION = SOURCE / "pet"
SIZE = 512


def prepare(source: Path, destination: Path, size: int = SIZE) -> tuple[int, int]:
    image = Image.open(source).convert("RGBA")
    bbox = image.getbbox()
    if bbox is None:
        raise ValueError(f"{source.name} está completamente vacío.")
    content = image.crop(bbox)
    side = max(content.size)
    padding = int(side * 0.06)
    canvas = Image.new("RGBA", (side + 2 * padding, side + 2 * padding), (0, 0, 0, 0))
    canvas.paste(
        content,
        (
            (canvas.width - content.width) // 2,
            (canvas.height - content.height) // 2,
        ),
        content,
    )
    resized = canvas.resize((size, size), Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    resized.save(destination, "PNG", optimize=True)
    return source.stat().st_size, destination.stat().st_size


def main() -> None:
    total_before = 0
    total_after = 0
    for source in sorted(SOURCE.glob("*.png")):
        destination = DESTINATION / source.name
        before, after = prepare(source, destination)
        total_before += before
        total_after += after
        print(f"{source.name:14s} {before / 1024:8.1f} KB -> {after / 1024:6.1f} KB")
    print(
        f"\nTotal: {total_before / 1024 / 1024:.2f} MB -> "
        f"{total_after / 1024 / 1024:.2f} MB"
    )
    print(f"Destino: {DESTINATION}")


if __name__ == "__main__":
    main()
