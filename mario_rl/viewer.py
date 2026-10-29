"""Native SDL window; avoids the legacy pyglet/OpenGL renderer on macOS."""
from __future__ import annotations

import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame
import numpy as np


class Viewer:
    def __init__(self, title="Mario RL | Apple Silicon", fps=15):
        pygame.init()
        self.screen = pygame.display.set_mode((1040, 824))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("Menlo", 17)
        self.small = pygame.font.SysFont("Menlo", 13)
        self.big = pygame.font.SysFont("Menlo", 25, bold=True)
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.paused = False
        self.reset_requested = False

    def poll(self):
        self.reset_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_p:
                    self.paused = not self.paused
                if event.key == pygame.K_r:
                    self.reset_requested = True
        return True

    def human_action(self):
        keys = pygame.key.get_pressed()
        right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        jump = keys[pygame.K_SPACE] or keys[pygame.K_z]
        run = keys[pygame.K_LSHIFT] or keys[pygame.K_x]
        if right:
            return 4 if jump and run else 2 if jump else 3 if run else 1
        return 6 if left else 5 if jump else 0

    def draw(self, frame, info, *, policy, step, episode, reward, device="cpu"):
        self.screen.fill((15, 20, 31))
        self.screen.blit(self.big.render("MARIO / REINFORCEMENT LEARNING", True, (240, 244, 249)), (24, 20))
        surface = pygame.surfarray.make_surface(np.asarray(frame).transpose(1, 0, 2))
        self.screen.blit(pygame.transform.scale(surface, (768, 720)), (24, 68))
        x = 814
        rows = [
            ("POLICY", policy), ("DEVICE", device.upper()),
            ("EPISODE", str(episode)), ("DECISIONS", str(step)),
            ("POSITION", str(info.get("x_pos", 0))),
            ("SCORE", str(info.get("score", 0))),
            ("COINS", str(info.get("coins", 0))),
            ("TIME", str(info.get("time", 400))),
            ("RETURN", f"{reward:.1f}"),
            ("STATUS", "CLEAR!" if info.get("flag_get") else "PAUSED" if self.paused else "PLAYING"),
        ]
        for i, (label, value) in enumerate(rows):
            y = 76 + i * 58
            self.screen.blit(self.small.render(label, True, (137, 153, 177)), (x, y))
            self.screen.blit(self.font.render(value[:21], True, (90, 226, 173)), (x, y + 21))
        for i, line in enumerate(["ARROWS / A,D : move", "SPACE / Z : jump", "SHIFT / X : run", "P pause   R restart", "ESC quit"]):
            self.screen.blit(self.small.render(line, True, (180, 191, 208)), (x, 678 + i * 23))
        level = f"{info.get('world', '?')}-{info.get('stage', '?')}"
        self.screen.blit(self.small.render(f"PPO + CNN | native arm64 NES emulator | World {level}", True, (137, 153, 177)), (24, 800))
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()
