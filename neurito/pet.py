"""Neurito en Pygame: el cerebro de NumPy convertido en mascota animada.

La red predice una acción a partir de tres necesidades. La mascota ejecuta esa
acción, la anima y actualiza sus necesidades; el bucle vuelve a empezar.

Ejecutar:

    python -m neurito.pet

Para una prueba sin ventana (útil en CI):

    python -m neurito.pet --headless --frames 120 --screenshot docs/neurito.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np

# El driver de video se elige antes de importar/initializar pygame.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

from .brain import ACTION_LABELS, ACTIONS, NeuralNet  # noqa: E402
from .train import MODEL_PATH, TrainConfig, train_default  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = PROJECT_ROOT / "assets" / "pet"
METADATA_PATH = PROJECT_ROOT / "models" / "neurito_metadata.json"

WIDTH, HEIGHT = 1120, 720
FPS = 60

BG_TOP = (255, 248, 244)
BG_BOTTOM = (255, 233, 240)
CARD = (255, 255, 255)
BORDER = (240, 220, 224)
INK = (52, 44, 57)
MUTED = (126, 113, 123)
CORAL = (232, 93, 117)
GREEN = (47, 163, 124)
BLUE = (76, 111, 255)
GOLD = (242, 169, 59)
PINK = (255, 196, 212)

ACTION_COLORS = {"dormir": GOLD, "comer": CORAL, "jugar": BLUE}
ACTION_DIALOGUE = {
    "dormir": "Tengo sueño...",
    "comer": "¡Tengo hambre!",
    "jugar": "¡Quiero jugar!",
}
NEED_COLORS = [GOLD, CORAL, BLUE]
NEED_NAMES = ["Energía", "Hambre", "Aburrimiento"]


def _font(size: int, bold: bool = False) -> pygame.font.Font:
    path = Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf")
    if path.exists():
        return pygame.font.Font(str(path), size)
    return pygame.font.SysFont("segoeui", size, bold=bold)


def _text(
    surface: pygame.Surface,
    value: str,
    position: tuple[float, float],
    size: int = 20,
    color: tuple[int, int, int] = INK,
    *,
    bold: bool = False,
    anchor: str = "topleft",
) -> pygame.Rect:
    image = _font(size, bold).render(value, True, color)
    rect = image.get_rect()
    setattr(rect, anchor, (int(position[0]), int(position[1])))
    surface.blit(image, rect)
    return rect


def _mix(color_a: tuple[int, int, int], color_b: tuple[int, int, int], amount: float):
    amount = max(0.0, min(1.0, amount))
    return tuple(int(a + (b - a) * amount) for a, b in zip(color_a, color_b))


def _card(surface: pygame.Surface, rect: pygame.Rect, radius: int = 22) -> None:
    shadow = pygame.Surface((rect.w + 16, rect.h + 16), pygame.SRCALPHA)
    pygame.draw.rect(shadow, (120, 80, 92, 22), (8, 9, rect.w, rect.h), border_radius=radius)
    surface.blit(shadow, (rect.x - 8, rect.y - 7))
    pygame.draw.rect(surface, CARD, rect, border_radius=radius)
    pygame.draw.rect(surface, BORDER, rect, width=1, border_radius=radius)


def _bar(
    surface: pygame.Surface,
    rect: pygame.Rect,
    value: float,
    color: tuple[int, int, int],
    background: tuple[int, int, int] = (244, 235, 237),
) -> None:
    pygame.draw.rect(surface, background, rect, border_radius=rect.h // 2)
    fill_width = int(rect.w * max(0.0, min(1.0, value)))
    if fill_width > 1:
        fill = pygame.Rect(rect.x, rect.y, fill_width, rect.h)
        pygame.draw.rect(surface, color, fill, border_radius=rect.h // 2)


def _heart(surface: pygame.Surface, x: float, y: float, size: float, color, alpha: int = 255) -> None:
    overlay = pygame.Surface((int(size * 3), int(size * 3)), pygame.SRCALPHA)
    cx, cy = size * 1.5, size * 1.3
    pygame.draw.circle(overlay, (*color, alpha), (int(cx - size * 0.42), int(cy)), int(size * 0.48))
    pygame.draw.circle(overlay, (*color, alpha), (int(cx + size * 0.42), int(cy)), int(size * 0.48))
    pygame.draw.polygon(
        overlay,
        (*color, alpha),
        [
            (cx - size * 0.9, cy + size * 0.1),
            (cx + size * 0.9, cy + size * 0.1),
            (cx, cy + size * 1.15),
        ],
    )
    surface.blit(overlay, (x - size * 1.5, y - size * 1.3))


class Particle:
    def __init__(self, x: float, y: float, color, *, kind: str = "heart", size: float = 7):
        self.x = x
        self.y = y
        self.vx = random.uniform(-14, 14)
        self.vy = random.uniform(-50, -26)
        self.life = 1.0
        self.color = color
        self.kind = kind
        self.size = size
        self.phase = random.uniform(0, math.tau)

    def update(self, dt: float) -> None:
        self.life -= dt * 0.9
        self.x += (self.vx + math.sin(self.phase + self.life * 6) * 9) * dt
        self.y += self.vy * dt
        self.vy += 10 * dt

    @property
    def alive(self) -> bool:
        return self.life > 0

    def draw(self, surface: pygame.Surface) -> None:
        alpha = int(255 * min(1.0, self.life * 1.4))
        if self.kind == "heart":
            _heart(surface, self.x, self.y, self.size, self.color, alpha)
        else:
            overlay = pygame.Surface((30, 30), pygame.SRCALPHA)
            pygame.draw.circle(overlay, (*self.color, alpha), (15, 15), int(self.size * 0.45))
            pygame.draw.circle(overlay, (255, 255, 255, alpha), (13, 12), int(self.size * 0.16))
            surface.blit(overlay, (self.x - 15, self.y - 15))


class Neurito:
    """Estado de la mascota: necesidades, predicción y animación."""

    def __init__(self, model: NeuralNet):
        self.model = model
        self.frames = self._load_frames()
        self.reset()

    @staticmethod
    def _load_frames() -> dict[str, list[pygame.Surface]]:
        names = {
            "neutral": ["neutral_1.png", "neutral_2.png", "neutral_3.png"],
            "dormir": ["dormir_1.png", "dormir_2.png"],
            "comer": ["comer_1.png", "comer_2.png"],
            "jugar": ["jugar_1.png", "jugar_2.png"],
        }
        frames: dict[str, list[pygame.Surface]] = {}
        size = (410, 410)
        for action, files in names.items():
            frames[action] = []
            for file in files:
                path = ASSET_DIR / file
                if not path.exists():
                    raise FileNotFoundError(
                        f"Falta {path}. Ejecuta: python tools/prepare_assets.py"
                    )
                image = pygame.image.load(str(path)).convert_alpha()
                frames[action].append(pygame.transform.smoothscale(image, size))
        return frames

    def reset(self) -> None:
        self.needs = np.array([0.35, 0.90, 0.25], dtype=float)
        self.action = "neutral"
        self.raw_action = "dormir"
        self.probabilities = np.array([1 / 3, 1 / 3, 1 / 3])
        self.hidden = np.zeros(self.model.hidden)
        self.time = 0.0
        self.think_timer = 0.0
        self.candidate = None
        self.candidate_time = 0.0
        self.paused = False
        self.speed = 1.0
        self.frame_index = 0
        self.frame_timer = 0.0
        self.particles: list[Particle] = []
        self.think()
        self.action = self.raw_action
        self.frame_index = 0

    def think(self) -> None:
        needs = np.clip(self.needs, 0.0, 1.0)
        probabilities, cache = self.model.forward(needs.reshape(1, -1), cache=True)
        self.probabilities = probabilities[0]
        self.hidden = cache["a1"][0]
        self.raw_action = ACTIONS[int(np.argmax(self.probabilities))]

    def _set_action(self, action: str) -> None:
        if action != self.action:
            self.action = action
            self.frame_index = 0
            self.frame_timer = 0.0

    def update(self, dt: float) -> None:
        if self.paused:
            return
        self.time += dt
        self.needs[0] -= 0.011 * dt
        self.needs[1] += 0.013 * dt
        self.needs[2] += 0.017 * dt

        if self.action == "dormir":
            self.needs[0] += 0.070 * dt
            self.needs[1] += 0.003 * dt
            self.needs[2] += 0.004 * dt
        elif self.action == "comer":
            self.needs[1] -= 0.080 * dt
            self.needs[2] += 0.009 * dt
        elif self.action == "jugar":
            self.needs[2] -= 0.090 * dt
            self.needs[0] -= 0.016 * dt
            self.needs[1] += 0.008 * dt

        self.needs = np.clip(self.needs, 0.0, 1.0)

        self.think_timer -= dt
        if self.think_timer <= 0:
            self.think()
            self.think_timer = 0.12

        if self.raw_action != self.action and self.action != "neutral":
            if self.raw_action == self.candidate:
                self.candidate_time += dt
            else:
                self.candidate = self.raw_action
                self.candidate_time = 0.0
            if self.candidate_time >= 0.35:
                self._set_action(self.raw_action)
                self.candidate = None
        elif self.action == "neutral":
            self._set_action(self.raw_action)

        self.frame_timer += dt
        duration = {"neutral": 0.62, "dormir": 1.25, "comer": 0.34, "jugar": 0.24}[
            self.action
        ]
        if self.frame_timer >= duration:
            self.frame_timer = 0.0
            self.frame_index = (self.frame_index + 1) % len(self.frames[self.action])

        for particle in self.particles:
            particle.update(dt)
        self.particles = [particle for particle in self.particles if particle.alive]

    def apply(self, kind: str) -> None:
        if kind == "comer":
            self.needs[1] = max(0.0, self.needs[1] - 0.34)
            self._burst(7, CORAL)
        elif kind == "jugar":
            self.needs[2] = max(0.0, self.needs[2] - 0.34)
            self.needs[0] = max(0.0, self.needs[0] - 0.06)
            self._burst(7, BLUE)
        elif kind == "dormir":
            self.needs[0] = min(1.0, self.needs[0] + 0.34)
            self._burst(6, GOLD)
        self.candidate = None
        self.think()

    def _burst(self, quantity: int, color) -> None:
        for _ in range(quantity):
            self.particles.append(
                Particle(
                    610 + random.uniform(-120, 120),
                    430 + random.uniform(-40, 30),
                    color,
                    kind=random.choice(["heart", "heart", "spark"]),
                    size=random.uniform(5, 10),
                )
            )

    def draw(self, surface: pygame.Surface) -> None:
        bob = {
            "neutral": math.sin(self.time * 1.8) * 4,
            "dormir": math.sin(self.time * 1.1) * 7,
            "comer": abs(math.sin(self.time * 5.2)) * 7,
            "jugar": -abs(math.sin(self.time * 4.6)) * 34,
        }[self.action]
        center = (610, 347 + bob)

        shadow = pygame.Surface((360, 70), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (120, 75, 90, 34), (32, 18, 296, 34))
        surface.blit(shadow, (430, 505))

        image = self.frames[self.action][self.frame_index]
        rect = image.get_rect(center=center)
        surface.blit(image, rect)

        # Efectos propios de cada acción.
        if self.action == "dormir":
            for index in range(3):
                phase = (self.time * 0.45 + index * 0.33) % 1.0
                alpha = int(220 * (1.0 - phase))
                z = _font(24 + index * 10, bold=True).render("z", True, GOLD)
                z.set_alpha(alpha)
                surface.blit(
                    z,
                    (rect.right - 60 + phase * 46, rect.top + 60 - phase * 95),
                )
        elif self.action == "comer":
            bowl = pygame.Rect(rect.right - 120, rect.bottom - 55, 105, 38)
            pygame.draw.ellipse(surface, (222, 135, 104), bowl)
            pygame.draw.ellipse(surface, (250, 198, 168), bowl.inflate(-8, -12))
            for index in range(4):
                angle = self.time * 5 + index * 1.7
                px = bowl.centerx + math.cos(angle) * (10 + index * 3)
                py = bowl.y + 8 + math.sin(angle * 1.3) * 5
                pygame.draw.circle(surface, (146, 91, 60), (int(px), int(py)), 4)
        elif self.action == "jugar":
            ball_x = rect.right - 55 + math.sin(self.time * 3.6) * 30
            ball_y = rect.bottom - 78 - abs(math.sin(self.time * 4.6)) * 95
            pygame.draw.circle(surface, CORAL, (int(ball_x), int(ball_y)), 18)
            pygame.draw.arc(
                surface,
                CARD,
                (int(ball_x) - 18, int(ball_y) - 18, 36, 36),
                0.2,
                2.4,
                3,
            )
            pygame.draw.circle(surface, (255, 255, 255), (int(ball_x) - 6, int(ball_y) - 7), 4)

        for particle in self.particles:
            particle.draw(surface)

        confidence = float(np.max(self.probabilities))
        color = ACTION_COLORS[self.action]
        badge = pygame.Rect(0, 0, 236, 34)
        badge.center = (610, 566)
        pygame.draw.rect(surface, color, badge, border_radius=17)
        label = ACTION_LABELS.get(self.action, "Neutral")
        _text(
            surface,
            f"{label.upper()}   ·   {confidence:.0%}",
            badge.center,
            17,
            (255, 255, 255),
            bold=True,
            anchor="center",
        )
        _text(surface, ACTION_DIALOGUE[self.action], (610, 536), 18, MUTED, anchor="center")


class App:
    def __init__(
        self,
        headless: bool = False,
        screenshot: Path | None = None,
        max_frames: int = 240,
    ):
        self.headless = headless
        self.screenshot = screenshot
        self.max_frames = max_frames
        self.model = self._load_model()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Neurito · una mascota con cerebro artificial")
        self.clock = pygame.time.Clock()
        self.pet = Neurito(self.model)
        self.background = self._make_background()
        self.stars = [
            {
                "x": random.uniform(20, WIDTH - 20),
                "y": random.uniform(80, HEIGHT - 20),
                "r": random.uniform(1.5, 4.0),
                "phase": random.uniform(0, math.tau),
                "speed": random.uniform(0.3, 1.0),
            }
            for _ in range(26)
        ]
        self.buttons = {
            "comer": pygame.Rect(414, 600, 116, 40),
            "jugar": pygame.Rect(542, 600, 116, 40),
            "dormir": pygame.Rect(670, 600, 116, 40),
        }
        self.mouse = (0, 0)
        self.metadata = self._load_metadata()
        self.running = True

    @staticmethod
    def _load_model() -> NeuralNet:
        if not MODEL_PATH.exists():
            print("No hay modelo guardado; entrenando uno nuevo...")
            train_default(TrainConfig(), save=True, verbose=False)
        model, _ = NeuralNet.load(MODEL_PATH)
        return model

    @staticmethod
    def _load_metadata() -> dict:
        if METADATA_PATH.exists():
            return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        return {}

    def _make_background(self) -> pygame.Surface:
        surface = pygame.Surface((WIDTH, HEIGHT))
        for y in range(HEIGHT):
            amount = y / (HEIGHT - 1)
            pygame.draw.line(surface, _mix(BG_TOP, BG_BOTTOM, amount), (0, y), (WIDTH, y))
        glow = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*PINK, 48), (910, 120), 230)
        pygame.draw.circle(glow, (190, 220, 255, 60), (180, 610), 260)
        pygame.draw.circle(glow, (255, 236, 190, 45), (1060, 620), 190)
        surface.blit(glow, (0, 0))
        return surface

    def draw_header(self) -> None:
        pygame.draw.circle(self.screen, CORAL, (55, 54), 21)
        _text(self.screen, "N", (55, 51), 24, (255, 255, 255), bold=True, anchor="center")
        _text(self.screen, "NEURITO", (88, 31), 34, INK, bold=True)
        _text(self.screen, "una mascota con cerebro artificial", (90, 71), 16, MUTED)

        accuracy = self.metadata.get("metrics", {}).get("test_accuracy")
        epochs = self.metadata.get("config", {}).get("epochs", 800)
        if accuracy is not None:
            status = f"precisión {accuracy:.1%}  ·  {epochs} épocas"
        else:
            status = "cerebro entrenado"
        _text(self.screen, status, (1088, 58), 14, MUTED, anchor="topright")
        pygame.draw.circle(self.screen, GREEN, (842, 50), 5)
        _text(self.screen, "EN LÍNEA", (854, 39), 13, GREEN, bold=True)

    def draw_needs(self) -> None:
        card = pygame.Rect(32, 112, 330, 330)
        _card(self.screen, card)
        _text(self.screen, "Necesidades", (54, 130), 23, INK, bold=True)
        _text(self.screen, "lo que siente Neurito", (54, 160), 14, MUTED)

        for index, (name, color) in enumerate(zip(NEED_NAMES, NEED_COLORS)):
            top = 190 + index * 78
            pygame.draw.circle(self.screen, color, (58, top + 13), 6)
            _text(self.screen, name, (72, top), 17, INK)
            _text(
                self.screen,
                f"{self.pet.needs[index] * 100:.0f}%",
                (342, top),
                17,
                color,
                bold=True,
                anchor="topright",
            )
            _bar(
                self.screen,
                pygame.Rect(54, top + 29, 274, 13),
                float(self.pet.needs[index]),
                color,
            )
        _text(self.screen, "los valores cambian con el tiempo", (54, 408), 12, MUTED)

    def draw_probabilities(self) -> None:
        card = pygame.Rect(32, 458, 330, 206)
        _card(self.screen, card)
        _text(self.screen, "Salidas Softmax", (54, 474), 21, INK, bold=True)
        best = int(np.argmax(self.pet.probabilities))
        for index, action in enumerate(ACTIONS):
            top = 510 + index * 44
            color = ACTION_COLORS[action]
            label = ACTION_LABELS[action]
            if index == best:
                pygame.draw.circle(self.screen, color, (58, top + 10), 4)
            _text(self.screen, label, (72, top - 2), 16, INK if index == best else MUTED)
            _bar(
                self.screen,
                pygame.Rect(154, top + 3, 160, 12),
                float(self.pet.probabilities[index]),
                color,
            )
            _text(
                self.screen,
                f"{self.pet.probabilities[index] * 100:.0f}%",
                (342, top - 2),
                15,
                color if index == best else MUTED,
                bold=index == best,
                anchor="topright",
            )

    def draw_brain(self) -> None:
        card = pygame.Rect(820, 112, 268, 552)
        _card(self.screen, card)
        _text(self.screen, "Cerebro", (842, 128), 23, INK, bold=True)
        _text(self.screen, "3 → 6 ReLU → 3 Softmax", (842, 158), 13, MUTED)

        net = pygame.Surface((card.w, 360), pygame.SRCALPHA)
        input_x, hidden_x, output_x = 44, 138, 226
        input_y = [90, 165, 240]
        hidden_y = [42 + index * 57 for index in range(self.model.hidden)]
        output_y = [90, 165, 240]

        max_w1 = float(np.max(np.abs(self.model.W1))) or 1.0
        max_w2 = float(np.max(np.abs(self.model.W2))) or 1.0
        for i, y1 in enumerate(input_y):
            for h, y2 in enumerate(hidden_y):
                weight = float(self.model.W1[i, h])
                color = CORAL if weight >= 0 else BLUE
                width = max(1, int(abs(weight) / max_w1 * 4))
                alpha = 35 + int(abs(weight) / max_w1 * 100)
                pygame.draw.line(net, (*color, alpha), (input_x, y1), (hidden_x, y2), width)
        for h, y1 in enumerate(hidden_y):
            for o, y2 in enumerate(output_y):
                weight = float(self.model.W2[h, o])
                color = CORAL if weight >= 0 else BLUE
                width = max(1, int(abs(weight) / max_w2 * 4))
                alpha = 35 + int(abs(weight) / max_w2 * 100)
                pygame.draw.line(net, (*color, alpha), (hidden_x, y1), (output_x, y2), width)

        for index, y in enumerate(input_y):
            pygame.draw.circle(net, NEED_COLORS[index], (input_x, y), 11)
            pygame.draw.circle(net, CARD, (input_x, y), 11, 2)
        activation_max = float(np.max(self.pet.hidden)) or 1.0
        for index, y in enumerate(hidden_y):
            amount = float(self.pet.hidden[index]) / activation_max
            fill = _mix((246, 224, 230), CORAL, amount)
            pygame.draw.circle(net, fill, (hidden_x, y), 9)
            pygame.draw.circle(net, (219, 159, 174), (hidden_x, y), 9, 1)
        for index, y in enumerate(output_y):
            color = ACTION_COLORS[ACTIONS[index]]
            pygame.draw.circle(net, color, (output_x, y), 11)
            pygame.draw.circle(net, CARD, (output_x, y), 11, 2)

        self.screen.blit(net, (card.x, 185))
        for index, y in enumerate(input_y):
            _text(self.screen, ["E", "H", "A"][index], (card.x + 12, 185 + y), 11, MUTED, anchor="midleft")
        for index, y in enumerate(output_y):
            _text(
                self.screen,
                ["D", "C", "J"][index],
                (card.x + 250, 185 + y),
                11,
                ACTION_COLORS[ACTIONS[index]],
                bold=True,
                anchor="midright",
            )

        _text(self.screen, "neuronas ocultas", (842, 528), 13, MUTED)
        max_activation = float(np.max(self.pet.hidden)) or 1.0
        for index in range(self.model.hidden):
            top = 550 + index * 15
            _text(self.screen, f"H{index + 1}", (842, top - 2), 10, MUTED)
            _bar(
                self.screen,
                pygame.Rect(868, top, 84, 8),
                float(self.pet.hidden[index]) / max_activation,
                _mix(CORAL, GOLD, index / max(1, self.model.hidden - 1)),
            )

    def draw_buttons(self) -> None:
        for action, rect in self.buttons.items():
            hover = rect.collidepoint(self.mouse)
            base = ACTION_COLORS[action]
            fill = _mix(base, (255, 255, 255), 0.12 if hover else 0.0)
            pygame.draw.rect(self.screen, fill, rect, border_radius=15)
            if hover:
                pygame.draw.rect(self.screen, CARD, rect, width=2, border_radius=15)
            labels = {"comer": "Comida", "jugar": "Jugar", "dormir": "Siesta"}
            _text(
                self.screen,
                labels[action],
                rect.center,
                16,
                (255, 255, 255),
                bold=True,
                anchor="center",
            )

    def draw_footer(self) -> None:
        pygame.draw.line(self.screen, BORDER, (32, 666), (WIDTH - 32, 666), 1)
        _text(
            self.screen,
            "E/D energía   ·   H/J hambre   ·   A/S aburrimiento   ·   ESPACIO pausa   ·   R reiniciar",
            (32, 678),
            13,
            MUTED,
        )
        _text(
            self.screen,
            "F comida   ·   P jugar   ·   N siesta   ·   +/- velocidad   ·   ESC salir",
            (32, 700),
            13,
            MUTED,
        )
        _text(
            self.screen,
            f"velocidad ×{self.pet.speed:.1f}",
            (1088, 700),
            13,
            MUTED,
            anchor="topright",
        )

    def draw_paused(self) -> None:
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((255, 248, 244, 150))
        self.screen.blit(overlay, (0, 0))
        _text(self.screen, "PAUSA", (WIDTH // 2, HEIGHT // 2 - 8), 42, INK, bold=True, anchor="center")
        _text(
            self.screen,
            "pulsa ESPACIO para continuar",
            (WIDTH // 2, HEIGHT // 2 + 34),
            17,
            MUTED,
            anchor="center",
        )

    def draw(self) -> None:
        self.screen.blit(self.background, (0, 0))
        for star in self.stars:
            alpha = 90 + int(50 * (0.5 + 0.5 * math.sin(self.pet.time * star["speed"] + star["phase"])))
            pygame.draw.circle(
                self.screen,
                (255, 214, 224),
                (int(star["x"]), int(star["y"])),
                int(star["r"]),
            )
        self.draw_header()
        self.draw_needs()
        self.draw_probabilities()
        self.draw_brain()
        self.pet.draw(self.screen)
        self.draw_buttons()
        self.draw_footer()
        if self.pet.paused:
            self.draw_paused()

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_SPACE:
                self.pet.paused = not self.pet.paused
            elif event.key == pygame.K_r:
                self.pet.reset()
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                self.pet.speed = min(4.0, self.pet.speed + 0.5)
            elif event.key == pygame.K_MINUS:
                self.pet.speed = max(0.5, self.pet.speed - 0.5)
            elif event.key == pygame.K_e:
                self.pet.needs[0] = min(1.0, self.pet.needs[0] + 0.1)
                self.pet.think()
            elif event.key == pygame.K_d:
                self.pet.needs[0] = max(0.0, self.pet.needs[0] - 0.1)
                self.pet.think()
            elif event.key == pygame.K_h:
                self.pet.needs[1] = min(1.0, self.pet.needs[1] + 0.1)
                self.pet.think()
            elif event.key == pygame.K_j:
                self.pet.needs[1] = max(0.0, self.pet.needs[1] - 0.1)
                self.pet.think()
            elif event.key == pygame.K_a:
                self.pet.needs[2] = min(1.0, self.pet.needs[2] + 0.1)
                self.pet.think()
            elif event.key == pygame.K_s:
                self.pet.needs[2] = max(0.0, self.pet.needs[2] - 0.1)
                self.pet.think()
            elif event.key == pygame.K_f:
                self.pet.apply("comer")
            elif event.key == pygame.K_p:
                self.pet.apply("jugar")
            elif event.key == pygame.K_n:
                self.pet.apply("dormir")
        elif event.type == pygame.MOUSEMOTION:
            self.mouse = event.pos
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for action, rect in self.buttons.items():
                if rect.collidepoint(event.pos):
                    self.pet.apply(action)

    def run(self) -> None:
        frames = 0
        while self.running:
            if self.headless:
                dt = 1.0 / FPS
                self.clock.tick(FPS)
            else:
                dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
            for event in pygame.event.get():
                self.handle_event(event)
            if not self.pet.paused:
                self.pet.update(dt * self.pet.speed)
            self.draw()
            pygame.display.flip()
            frames += 1
            if self.headless and frames >= self.max_frames:
                break

        if self.screenshot:
            self.screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(self.screen, str(self.screenshot))
            print(f"Captura guardada en {self.screenshot}")
        pygame.quit()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Abre a Neurito en una ventana.")
    parser.add_argument("--headless", action="store_true", help="Sin ventana; útil para pruebas.")
    parser.add_argument("--frames", type=int, default=120, help="Frames en modo headless.")
    parser.add_argument("--screenshot", type=Path, default=None, help="Ruta de la captura final.")
    args = parser.parse_args(argv)

    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

    pygame.init()
    pygame.font.init()
    app = App(
        headless=args.headless,
        screenshot=args.screenshot,
        max_frames=args.frames,
    )
    app.run()


if __name__ == "__main__":
    main()
