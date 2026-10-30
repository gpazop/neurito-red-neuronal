"""Gráficas del proyecto dibujadas con Pillow, sin depender de matplotlib."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BACKGROUND = "#FFF8F3"
CARD = "#FFFFFF"
INK = "#332B36"
MUTED = "#7C6F7A"
GRID = "#EADFD9"
BORDER = "#E7D8D2"
CORAL = "#E85D75"
GREEN = "#2FA37C"
BLUE = "#4C6FFF"
GOLD = "#F2A93B"
PURPLE = "#8B5CF6"
TEAL = "#1F9D9D"
PALETTE = [CORAL, GREEN, BLUE, GOLD, PURPLE, TEAL, "#C05299"]

FONT_REGULAR = Path("C:/Windows/Fonts/segoeui.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/segoeuib.ttf")
FONT_FALLBACK = Path("C:/Windows/Fonts/arial.ttf")

WIDTH = 1400
HEIGHT = 820


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [FONT_BOLD if bold else FONT_REGULAR, FONT_BOLD, FONT_FALLBACK]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def _new_figure(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((28, 24, WIDTH - 28, HEIGHT - 24), radius=26, fill=CARD, outline=BORDER, width=2)
    draw.text((64, 50), title, fill=INK, font=_font(38, bold=True))
    if subtitle:
        draw.text((64, 98), subtitle, fill=MUTED, font=_font(22))
    return image, draw


def _plot_area(draw: ImageDraw.ImageDraw):
    left, top, right, bottom = 104, 158, WIDTH - 64, HEIGHT - 104
    draw.rectangle((left, top, right, bottom), fill="#FFFCFA", outline=BORDER, width=1)
    return left, top, right, bottom


def _format_number(value: float, decimals: int = 2) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{decimals}f}".replace(".", ",")


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    entries: Sequence[tuple[str, str]],
    right: int,
    top: int,
    columns: int = 2,
    *,
    left: int | None = None,
) -> None:
    if not entries:
        return
    font = _font(19)
    row_height = 28
    rows = int(np.ceil(len(entries) / columns))
    column_width = 210
    width = columns * column_width
    height = rows * row_height + 18
    if left is None:
        left = right - width - 16
    draw.rounded_rectangle(
        (left, top, left + width, top + height), radius=12, fill=(255, 255, 255, 245), outline=BORDER
    )
    for index, (name, color) in enumerate(entries):
        column = index // rows
        row = index % rows
        x = left + 16 + column * column_width
        y = top + 12 + row * row_height
        draw.rounded_rectangle((x, y + 7, x + 20, y + 16), radius=4, fill=color)
        draw.text((x + 29, y), name, fill=INK, font=font)


def _nice_axis(low: float, high: float, target_ticks: int = 6) -> tuple[float, float, np.ndarray]:
    """Extiende un rango a extremos "redondos" y devuelve sus marcas."""

    if not np.isfinite(low) or not np.isfinite(high):
        low, high = 0.0, 1.0
    if high <= low:
        high = low + 1.0
    raw_step = (high - low) / max(1, target_ticks - 1)
    exponent = np.floor(np.log10(raw_step))
    fraction = raw_step / (10.0**exponent)
    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
        if fraction <= candidate + 1e-12:
            step = candidate * (10.0**exponent)
            break
    else:
        step = 10.0 ** (exponent + 1)
    start = np.floor(low / step) * step
    end = np.ceil(high / step) * step
    ticks = np.arange(start, end + step * 0.5, step)
    return float(start), float(end), ticks


def line_chart(
    path: str | Path,
    *,
    title: str,
    subtitle: str = "",
    series: Sequence[dict],
    xlabel: str,
    ylabel: str,
    y_min: float | None = None,
    y_max: float | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    y_decimals: int = 2,
    x_decimals: int = 2,
    vlines: Sequence[tuple[float, str, str]] = (),
    annotation: str = "",
    markers: bool = False,
) -> Path:
    image, draw = _new_figure(title, subtitle)
    left, top, right, bottom = _plot_area(draw)

    all_x = np.concatenate([np.asarray(item["x"], dtype=float) for item in series])
    all_y = np.concatenate([np.asarray(item["y"], dtype=float) for item in series])
    finite = np.isfinite(all_y)
    if not np.any(finite):
        all_y = np.array([0.0, 1.0])
        finite = np.array([True, True])
    low_y = float(np.min(all_y[finite])) if y_min is None else y_min
    high_y = float(np.max(all_y[finite])) if y_max is None else y_max
    low_x = float(np.min(all_x)) if x_min is None else x_min
    high_x = float(np.max(all_x)) if x_max is None else x_max
    if high_y <= low_y:
        high_y = low_y + 1.0
    if high_x <= low_x:
        high_x = low_x + 1.0
    low_y, high_y, y_ticks = _nice_axis(low_y, high_y, 6)
    low_x, high_x, x_ticks = _nice_axis(low_x, high_x, 7)

    def px(x: float) -> float:
        return left + (x - low_x) / (high_x - low_x) * (right - left)

    def py(y: float) -> float:
        return bottom - (y - low_y) / (high_y - low_y) * (bottom - top)

    tick_font = _font(18)
    for value in y_ticks:
        y = py(value)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((left - 14, y - 11), _format_number(float(value), y_decimals), fill=MUTED, font=tick_font, anchor="ra")
    for value in x_ticks:
        x = px(value)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        draw.text((x, bottom + 14), _format_number(float(value), x_decimals), fill=MUTED, font=tick_font, anchor="ma")

    draw.text(((left + right) / 2, bottom + 54), xlabel, fill=MUTED, font=_font(21), anchor="ma")
    y_label_image = Image.new("RGBA", (400, 36), (0, 0, 0, 0))
    y_draw = ImageDraw.Draw(y_label_image)
    y_draw.text((200, 18), ylabel, fill=MUTED, font=_font(21), anchor="mm")
    rotated = y_label_image.rotate(90, expand=True)
    image.paste(rotated, (10, int((top + bottom) / 2) - 200), rotated)

    for value, label, color in vlines:
        if low_x <= value <= high_x:
            x = px(value)
            for y in range(top, bottom, 12):
                draw.line((x, y, x, min(y + 6, bottom)), fill=color, width=2)
            draw.text((x - 7, top + 8), label, fill=color, font=_font(17), anchor="ra")

    for item in series:
        color = item.get("color", CORAL)
        xs = np.asarray(item["x"], dtype=float)
        ys = np.asarray(item["y"], dtype=float)
        points = [
            (px(float(x)), py(float(y)))
            for x, y in zip(xs, ys)
            if np.isfinite(y) and low_x <= x <= high_x
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=4, joint="curve")
        if markers:
            for point in points[:: max(1, len(points) // 14)]:
                draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=CARD, outline=color, width=3)

    _draw_legend(
        draw,
        [(item["name"], item.get("color", CORAL)) for item in series],
        right,
        top + 14,
        columns=2 if len(series) > 1 else 1,
    )
    if annotation:
        draw.text((right - 20, bottom - 32), annotation, fill=MUTED, font=_font(18), anchor="ra")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def bar_chart(
    path: str | Path,
    *,
    title: str,
    subtitle: str = "",
    categories: Sequence[str],
    series: Sequence[dict],
    ylabel: str,
    y_max: float | None = None,
    value_format: str = "{:.2f}",
    legend: bool = True,
    legend_side: str = "right",
    hlines: Sequence[tuple[float, str, str]] = (),
) -> Path:
    image, draw = _new_figure(title, subtitle)
    left, top, right, bottom = _plot_area(draw)

    upper_candidates = []
    for item in series:
        values = np.asarray(item["values"], dtype=float)
        errors = np.asarray(item.get("errors", np.zeros_like(values)), dtype=float)
        upper_candidates.append(float(np.max(values + errors)))
    highest = max(upper_candidates) if y_max is None else float(y_max)
    axis_low, axis_high, ticks = _nice_axis(0.0, max(highest, 1e-9), 6)
    high = axis_high

    def py(value: float) -> float:
        return bottom - value / high * (bottom - top)

    tick_font = _font(18)
    for value in ticks:
        y = py(float(value))
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((left - 14, y - 11), _format_number(float(value)), fill=MUTED, font=tick_font, anchor="ra")

    for value, label, color in hlines:
        y = py(value)
        for x in range(left, right, 14):
            draw.line((x, y, min(x + 7, right), y), fill=color, width=2)
        draw.text((left + 8, y - 8), label, fill=color, font=_font(17), anchor="lb")

    n_groups = len(categories)
    n_series = len(series)
    group_width = (right - left) / n_groups
    bar_width = group_width * 0.68 / n_series
    for group_index, category in enumerate(categories):
        center = left + group_width * (group_index + 0.5)
        for series_index, item in enumerate(series):
            value = float(item["values"][group_index])
            x0 = center - (n_series * bar_width) / 2 + series_index * bar_width
            x1 = x0 + bar_width - 4
            color = item.get("color", PALETTE[series_index % len(PALETTE)])
            draw.rounded_rectangle((x0, py(min(value, high)), x1, bottom), radius=7, fill=color)
            error = item.get("errors", [0] * n_groups)[group_index]
            if error:
                error = float(error)
                center_x = (x0 + x1) / 2
                y_low = py(max(0.0, value - error))
                y_high = py(min(high, value + error))
                draw.line((center_x, y_low, center_x, y_high), fill=INK, width=2)
                draw.line((center_x - 7, y_high, center_x + 7, y_high), fill=INK, width=2)
                draw.line((center_x - 7, y_low, center_x + 7, y_low), fill=INK, width=2)
            label = value_format.format(value)
            if value > high * 0.78:
                draw.text(
                    ((x0 + x1) / 2, py(value) + 9),
                    label,
                    fill=CARD,
                    font=_font(18, bold=True),
                    anchor="ma",
                )
            else:
                font = _font(18, bold=True)
                position = ((x0 + x1) / 2, py(value) - 9)
                box = draw.textbbox(position, label, font=font, anchor="mb")
                draw.rectangle(
                    (box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2),
                    fill=CARD,
                )
                draw.text(
                    position,
                    label,
                    fill=INK,
                    font=font,
                    anchor="mb",
                )
        draw.text((center, bottom + 16), str(category), fill=INK, font=_font(21), anchor="ma")

    draw.text(((left + right) / 2, bottom + 58), "categoría", fill=MUTED, font=_font(20), anchor="ma")
    y_label_image = Image.new("RGBA", (400, 36), (0, 0, 0, 0))
    y_draw = ImageDraw.Draw(y_label_image)
    y_draw.text((200, 18), ylabel, fill=MUTED, font=_font(20), anchor="mm")
    rotated = y_label_image.rotate(90, expand=True)
    image.paste(rotated, (10, int((top + bottom) / 2) - 200), rotated)

    if legend and len(series) > 1:
        legend_left = left + 18 if legend_side == "left" else None
        _draw_legend(
            draw,
            [(item["name"], item.get("color", PALETTE[index])) for index, item in enumerate(series)],
            right,
            top + 14,
            columns=1,
            left=legend_left,
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def probability_chart(
    path: str | Path,
    *,
    title: str,
    subtitle: str,
    x: np.ndarray,
    probabilities: np.ndarray,
    xlabel: str,
    action_names: Sequence[str],
    colors: Sequence[str],
    teacher_thresholds: Sequence[float] = (),
    annotation: str = "",
) -> Path:
    series = [
        {"name": name, "x": x, "y": probabilities[:, index], "color": colors[index]}
        for index, name in enumerate(action_names)
    ]
    vlines = [
        (threshold, f"profesor {threshold:.2f}", "#9B8F99") for threshold in teacher_thresholds
    ]
    span = float(np.max(x) - np.min(x))
    return line_chart(
        path,
        title=title,
        subtitle=subtitle,
        series=series,
        xlabel=xlabel,
        ylabel="probabilidad de Softmax",
        y_min=0.0,
        y_max=1.0,
        x_min=float(np.min(x)),
        x_max=float(np.max(x)),
        y_decimals=2,
        x_decimals=3 if span < 0.05 else 2,
        vlines=vlines,
        annotation=annotation,
    )


def heatmap(
    path: str | Path,
    *,
    title: str,
    subtitle: str,
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    row_title: str,
    col_title: str,
    integer: bool = True,
    color: tuple[int, int, int] = (232, 93, 117),
) -> Path:
    image, draw = _new_figure(title, subtitle)
    matrix = np.asarray(matrix, dtype=float)
    rows, cols = matrix.shape
    left, top = 220, 190
    cell_w = (WIDTH - left - 90) / cols
    cell_h = (HEIGHT - top - 110) / rows
    maximum = max(float(matrix.max()), 1.0)
    for row in range(rows):
        for col in range(cols):
            intensity = matrix[row, col] / maximum
            fill = tuple(
                int(255 + (channel - 255) * intensity * 0.88) for channel in color
            )
            x0 = left + col * cell_w
            y0 = top + row * cell_h
            draw.rounded_rectangle(
                (x0 + 4, y0 + 4, x0 + cell_w - 4, y0 + cell_h - 4),
                radius=12,
                fill=fill,
                outline=BORDER,
            )
            text = f"{int(matrix[row, col])}" if integer else f"{matrix[row, col]:.2f}"
            draw.text(
                (x0 + cell_w / 2, y0 + cell_h / 2),
                text,
                fill=INK if intensity < 0.62 else CARD,
                font=_font(30, bold=True),
                anchor="mm",
            )
    for index, label in enumerate(row_labels):
        draw.text((left - 18, top + cell_h * (index + 0.5)), label, fill=INK, font=_font(23), anchor="rm")
    for index, label in enumerate(col_labels):
        draw.text(
            (left + cell_w * (index + 0.5), top + rows * cell_h + 18),
            label,
            fill=INK,
            font=_font(21),
            anchor="ma",
        )
    draw.text((left + cols * cell_w / 2, top + rows * cell_h + 62), col_title, fill=MUTED, font=_font(20), anchor="ma")
    draw.text((left - 18, top - 18), row_title, fill=MUTED, font=_font(19), anchor="rb")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def combine_horizontal(
    paths: Sequence[str | Path],
    destination: str | Path,
    *,
    gap: int = 22,
    background: str = BACKGROUND,
) -> Path:
    images = [Image.open(path).convert("RGB") for path in paths]
    height = max(image.height for image in images)
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), background)
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width + gap
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return destination


def combine_vertical(
    paths: Sequence[str | Path],
    destination: str | Path,
    *,
    gap: int = 22,
    background: str = BACKGROUND,
) -> Path:
    images = [Image.open(path).convert("RGB") for path in paths]
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), background)
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height + gap
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return destination


def architecture_diagram(model, path: str | Path, hidden_labels: Sequence[str] | None = None) -> Path:
    """Dibuja la red 3 -> H -> 3 usando los pesos entrenados."""

    image = Image.new("RGB", (1500, 820), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((24, 22, 1476, 798), radius=28, fill=CARD, outline=BORDER, width=2)
    draw.text((60, 48), "El cerebro de Neurito", fill=INK, font=_font(40, bold=True))
    draw.text(
        (60, 98),
        "Cada línea es un peso; el grosor indica su magnitud y el color su signo.",
        fill=MUTED,
        font=_font(22),
    )

    x_input, x_hidden, x_output = 260, 750, 1240
    y_center = 405
    input_gap = 135
    hidden_gap = 82
    output_gap = 135
    input_y = [y_center + (index - 1) * input_gap for index in range(3)]
    hidden_y = [
        y_center + (index - (model.hidden - 1) / 2) * hidden_gap
        for index in range(model.hidden)
    ]
    output_y = [y_center + (index - 1) * output_gap for index in range(3)]

    max_w1 = float(np.max(np.abs(model.W1))) or 1.0
    max_w2 = float(np.max(np.abs(model.W2))) or 1.0
    for input_index, y1 in enumerate(input_y):
        for hidden_index, y2 in enumerate(hidden_y):
            weight = float(model.W1[input_index, hidden_index])
            color = CORAL if weight >= 0 else BLUE
            width = max(1, int(abs(weight) / max_w1 * 7))
            draw.line((x_input, y1, x_hidden, y2), fill=color, width=width)
    for hidden_index, y1 in enumerate(hidden_y):
        for output_index, y2 in enumerate(output_y):
            weight = float(model.W2[hidden_index, output_index])
            color = CORAL if weight >= 0 else BLUE
            width = max(1, int(abs(weight) / max_w2 * 7))
            draw.line((x_hidden, y1, x_output, y2), fill=color, width=width)

    input_labels = ["Energía", "Hambre", "Aburrimiento"]
    output_labels = ["Dormir", "Comer", "Jugar"]
    node_colors = [GOLD, CORAL, BLUE]
    for index, y in enumerate(input_y):
        draw.ellipse((x_input - 30, y - 30, x_input + 30, y + 30), fill=node_colors[index], outline=CARD, width=4)
        draw.text((x_input - 48, y), input_labels[index], fill=INK, font=_font(22), anchor="rm")

    for index, y in enumerate(hidden_y):
        draw.ellipse((x_hidden - 24, y - 24, x_hidden + 24, y + 24), fill="#F6D9DF", outline=CORAL, width=3)
        draw.text((x_hidden, y), "ReLU", fill=INK, font=_font(13), anchor="mm")

    for index, y in enumerate(output_y):
        draw.ellipse((x_output - 30, y - 30, x_output + 30, y + 30), fill=node_colors[index], outline=CARD, width=4)
        draw.text((x_output + 48, y), output_labels[index], fill=INK, font=_font(22), anchor="lm")

    draw.text((x_input, 635), "3 necesidades", fill=MUTED, font=_font(21), anchor="ma")
    draw.text((x_hidden, 635), f"{model.hidden} neuronas ocultas", fill=MUTED, font=_font(21), anchor="ma")
    draw.text((x_output, 635), "3 acciones", fill=MUTED, font=_font(21), anchor="ma")
    draw.rounded_rectangle((255, 686, 385, 714), radius=8, fill=CORAL)
    draw.text((398, 686), "peso positivo", fill=MUTED, font=_font(18), anchor="lm")
    draw.rounded_rectangle((620, 686, 750, 714), radius=8, fill=BLUE)
    draw.text((763, 686), "peso negativo", fill=MUTED, font=_font(18), anchor="lm")
    draw.text(
        (750, 752),
        "salida = Softmax(ReLU(necesidades · W1 + b1) · W2 + b2)",
        fill=INK,
        font=_font(22),
        anchor="ma",
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
