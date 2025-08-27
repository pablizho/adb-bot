#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADB Scenario Bot (record & play) + basic Match-3 move (heuristic)

Dependencies:
    pip install opencv-python numpy pyyaml

Requires:
    - Android device with USB debugging enabled
    - adb in PATH

CLI:
    python adb_bot.py record --out scenario.yaml [--serial SERIAL]
      - OpenCV window:
         * LMB click           -> record TAP at cursor
         * LMB drag            -> record SWIPE from down to up position
         * Keys 1..9           -> record WAIT of N seconds
         * Key 'i'             -> select rect to save template + wait_image step
         * Key 's'             -> save current scenario to --out and exit
         * Key 'c'             -> clear last step (undo)
         * Key 'q'             -> quit without saving

    python adb_bot.py play --in scenario.yaml [--serial SERIAL]

    python adb_bot.py calibrate --rows 9 --cols 9 --out grid.yaml [--serial SERIAL]
      - Click TL cell center then BR cell center in screenshot

    python adb_bot.py match3 --grid grid.yaml [--serial SERIAL] [--mode swipe|tap]

Scenario YAML examples:
---
- type: tap
  at: [0.512, 0.736]          # normalized coords (0..1)
- type: swipe
  from: [0.3, 0.7]
  to: [0.8, 0.7]
  ms: 180
- type: wait
  seconds: 2.0
- type: wait_image
  template: templates/ok_btn.png
  threshold: 0.92
  timeout: 10.0
- type: match3_once
  grid: grid.yaml
  mode: swipe

Grid YAML (calibrated):
---
rows: 9
cols: 9
top_left: [120, 420]
bot_right: [960, 1260]

Notes:
- Match-3 is heuristic: color clustering via HSV distance; may require 'tol' tuning.
- For multi-resolution safety, taps/swipes are stored normalized. On play, mapped to current device res.
- wait_image uses cv2.matchTemplate; threshold ∈ [0,1]; template path relative to current dir.
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import json
import math
import subprocess
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional

import cv2
import numpy as np
import yaml
import av, threading


# рядом с импортами
def _default_grid_path():
    return "templates/grid.yaml" if os.path.exists("templates/grid.yaml") else "grid.yaml"


class ScreenStream:
    def __init__(self, adb: ADB, bitrate="8M"):
        self.adb = adb
        self.bitrate = bitrate
        self.proc = None
        self.container = None
        self.frame = None
        self._stop = threading.Event()
        self.t = None

    def start(self):
        args = ["adb"]
        if self.adb.serial:
            args += ["-s", self.adb.serial]
        args += ["exec-out", "screenrecord", "--output-format=h264", "--bit-rate", self.bitrate, "-"]
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE)
        # !!! больше НЕ вызываем av.open здесь
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        try:
            # !!! теперь открываем контейнер в фоне, чтобы не блокировать Tkinter
            self.container = av.open(self.proc.stdout, format="h264")
            for packet in self.container.demux(video=0):
                if self._stop.is_set():
                    break
                for frm in packet.decode():
                    self.frame = frm.to_ndarray(format="bgr24")
        except Exception as e:
            print("[stream] stopped:", e)
        finally:
            try:
                if self.container:
                    self.container.close()
            except Exception:
                pass

    def read(self):
        return self.frame

    def stop(self):
        self._stop.set()
        try:
            if self.container:
                self.container.close()
        except Exception:
            pass
        if self.proc:
            try:
                self.proc.kill()
            except Exception:
                pass
        if self.t:
            self.t.join(timeout=1)

def run_mirror(adb: "ADB", scale: float = 1.0):
    """
    Отдельное HighGUI-окно для трансляции.
    Масштаб по умолчанию 1:1 (без ресайза => без потери качества).
    Клавиши: ESC — выйти; '+'/'=' — увеличить; '-' — уменьшить; '1' — 1:1.
    """
    win = "ADB Mirror"
    stream = ScreenStream(adb, bitrate="8M")
    stream.start()
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cur_scale = float(scale)
        while True:
            # если окно закрыли крестиком
            try:
                vis = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE)
                if vis < 1:
                    break
            except cv2.error:
                break

            frm = stream.read()
            if frm is not None:
                if abs(cur_scale - 1.0) < 1e-3:
                    disp = frm
                else:
                    h, w = frm.shape[:2]
                    nw, nh = max(1, int(w * cur_scale)), max(1, int(h * cur_scale))
                    # внизмасштаб — INTER_AREA (качество), вверхмасштаб — INTER_NEAREST (без «мыла»)
                    inter = cv2.INTER_AREA if cur_scale < 1.0 else cv2.INTER_NEAREST
                    disp = cv2.resize(frm, (nw, nh), interpolation=inter)
                cv2.imshow(win, disp)

            key = (cv2.waitKey(1) & 0xFF)
            if key == 27:  # ESC
                break
            elif key in (ord('+'), ord('=')):
                cur_scale = min(4.0, cur_scale * 1.1)
            elif key == ord('-'):
                cur_scale = max(0.1, cur_scale / 1.1)
            elif key == ord('1'):
                cur_scale = 1.0

    except Exception as e:
        print(f"[mirror] error: {e}")
    finally:
        try:
            stream.stop()
        except Exception:
            pass
        try:
            cv2.destroyWindow(win)
        except Exception:
            pass


# ---------------------------- ADB utils ----------------------------

class ADB:
    def __init__(self, serial: Optional[str] = None):
        self.serial = serial

    def _adb(self, *args: str, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess:
        cmd = ["adb"]
        if self.serial:
            cmd += ["-s", self.serial]
        cmd += list(args)
        return subprocess.run(cmd, check=check, capture_output=capture_output)

    def tap(self, x: int, y: int):
        self._adb("shell", "input", "tap", str(x), str(y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, ms: int = 200):
        self._adb("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(ms))

    def screencap(self) -> np.ndarray:
        # Use exec-out to avoid temp file
        p = self._adb("exec-out", "screencap", "-p", capture_output=True)
        img = np.frombuffer(p.stdout, np.uint8)
        frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Failed to decode screencap; ensure device is unlocked and connected.")
        return frame

    def resolution(self) -> Tuple[int, int]:
        # Parse from wm size
        p = self._adb("shell", "wm", "size", capture_output=True)
        out = p.stdout.decode("utf-8", errors="ignore").strip()
        # Example: Physical size: 1080x2400
        for token in out.split():
            if "x" in token and token[0].isdigit():
                w, h = token.split("x")
                return int(w), int(h)
        # Fallback via screencap
        frame = self.screencap()
        h, w = frame.shape[:2]
        return w, h

# ---------------------------- Geometry helpers ----------------------------

def norm_to_px(norm_xy: Tuple[float, float], res: Tuple[int,int]) -> Tuple[int,int]:
    nx, ny = norm_xy
    w, h = res
    x = max(0, min(w-1, int(round(nx * w))))
    y = max(0, min(h-1, int(round(ny * h))))
    return x, y

# ---------------------------- Recorder ----------------------------

@dataclass
class Action:
    type: str
    data: Dict[str, Any]

class Recorder:
    def __init__(self, adb: ADB, out_path: str, mirror_live: bool = False):
        self.adb = adb
        self.out_path = out_path
        self.actions: List[Action] = []
        self.drag_start: Optional[Tuple[int,int]] = None
        self.display_scale = 1.0
        self.frame: Optional[np.ndarray] = None
        self.res: Tuple[int,int] = self.adb.resolution()
        self.window = "ADB Recorder"
        os.makedirs("templates", exist_ok=True)
        self.mirror_live = mirror_live


    def _push_action(self, typ: str, **kwargs):
        self.actions.append(Action(typ, kwargs))
        print(f"+ {typ}: {kwargs}")
        if self.mirror_live:
            try:
                if typ == "tap":
                    x, y = norm_to_px(tuple(kwargs["at"]), self.res)
                    self.adb.tap(x, y)
                elif typ == "swipe":
                    x1, y1 = norm_to_px(tuple(kwargs["from"]), self.res)
                    x2, y2 = norm_to_px(tuple(kwargs["to"]), self.res)
                    ms = int(kwargs.get("ms", 180))
                    self.adb.swipe(x1, y1, x2, y2, ms)
                elif typ == "wait":
                    time.sleep(float(kwargs.get("seconds", 1.0)))
                # wait_image во время записи только сохраняем (не исполняем)
            except Exception as e:
                print(f"[live] error: {e}")


    def _save(self):
        # Convert to YAML-friendly list
        steps: List[Dict[str,Any]] = []
        for a in self.actions:
            if a.type == "tap":
                steps.append({"type":"tap", "at": a.data["at"]})
            elif a.type == "swipe":
                steps.append({"type":"swipe", "from": a.data["from"], "to": a.data["to"], "ms": a.data["ms"]})
            elif a.type == "wait":
                steps.append({"type":"wait", "seconds": a.data["seconds"]})
            elif a.type == "wait_image":
                steps.append({"type":"wait_image", "template": a.data["template"], "threshold": a.data["threshold"], "timeout": a.data["timeout"]})
            elif a.type == "match3_once":
                steps.append({"type":"match3_once", "grid": a.data["grid"], "mode": a.data.get("mode","swipe")})
            else:
                steps.append({"type": a.type, **a.data})
        with open(self.out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(steps, f, allow_unicode=True, sort_keys=False)
        print(f"Saved: {self.out_path} ({len(steps)} steps)")

    def _on_mouse(self, event, x, y, flags, userdata):
        if self.frame is None:
            return
        # Map window coords to original frame coords
        scale = self.display_scale
        fx, fy = int(x/scale), int(y/scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (fx, fy)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag_start is None:
                return
            sx, sy = self.drag_start
            ex, ey = fx, fy
            self.drag_start = None
            # small movement -> tap, else swipe
            if abs(ex - sx) < 8 and abs(ey - sy) < 8:
                # tap: store normalized
                nx = sx / self.frame.shape[1]
                ny = sy / self.frame.shape[0]
                self._push_action("tap", at=[round(nx,4), round(ny,4)])
            else:
                n1x = sx / self.frame.shape[1]
                n1y = sy / self.frame.shape[0]
                n2x = ex / self.frame.shape[1]
                n2y = ey / self.frame.shape[0]
                self._push_action("swipe", **{"from":[round(n1x,4), round(n1y,4)], "to":[round(n2x,4), round(n2y,4)], "ms":180})

    def _select_rect(self) -> Optional[Tuple[int,int,int,int]]:
        print("[wait_image] Drag a rectangle with mouse (press and release LMB). Press ESC to cancel.")
        rs = self.display_scale
        drawing = {"start": None, "end": None}

        def cb(ev, x, y, flags, ud):
            if ev == cv2.EVENT_LBUTTONDOWN:
                drawing["start"] = (int(x/rs), int(y/rs))
                drawing["end"] = None
            elif ev == cv2.EVENT_MOUSEMOVE and drawing["start"] is not None:
                drawing["end"] = (int(x/rs), int(y/rs))
            elif ev == cv2.EVENT_LBUTTONUP and drawing["start"] is not None:
                drawing["end"] = (int(x/rs), int(y/rs))

        cv2.setMouseCallback(self.window, cb)
        try:
            while True:
                disp = self.frame.copy()
                if drawing["start"] and drawing["end"]:
                    x1,y1 = drawing["start"]; x2,y2 = drawing["end"]
                    x1,x2 = sorted([x1,x2]); y1,y2 = sorted([y1,y2])
                    cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,0), 2)
                disp_small, self.display_scale = fit_to_screen(disp)
                cv2.imshow(self.window, disp_small)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC
                    return None
                if key == ord('s') and drawing["start"] and drawing["end"]:
                    # нормализуем/проверяем прямоугольник
                    x1,y1 = drawing["start"]; x2,y2 = drawing["end"]
                    x1,x2 = sorted([x1,x2]); y1,y2 = sorted([y1,y2])
                    H, W = self.frame.shape[:2]
                    x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W))
                    y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H))
                    # требуем минимальный размер, иначе повторяем выделение
                    if (x2 - x1) < 5 or (y2 - y1) < 5:
                        print(f"[wait_image] selection too small ({x2-x1}x{y2-y1}), drag larger and press 's'")
                        continue
                    return x1, y1, x2, y2
        finally:
            cv2.setMouseCallback(self.window, self._on_mouse)




    def run(self):
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._on_mouse)
        print("Recorder controls: LMB tap/drag, keys 1..9=wait",
             "i=wait_image, a=match3_auto, f=autoplay_start, g=autoplay_stop",
             "m=match3_once, u=autoplay_until_image, c=undo, s=save&exit, q=quit")

        # стартуем стрим
        stream = ScreenStream(self.adb, bitrate="8M")
        stream.start()
        try:
            while True:
                # читаем кадр из стрима и показываем
                frm = stream.read()
                if frm is not None:
                    self.frame = frm
                    disp, self.display_scale = fit_to_screen(self.frame)
                    cv2.imshow(self.window, disp)

                key = cv2.waitKey(1) & 0xFF
                if key == 255:  # нет нажатий
                    continue

                if key in [ord(str(d)) for d in range(1,10)]:
                    seconds = int(chr(key))
                    self._push_action("wait", seconds=float(seconds))

                elif key == ord('i'):
                    if self.frame is None:
                        continue
                    rect = self._select_rect()
                    cv2.setMouseCallback(self.window, self._on_mouse)  # страховка
                    if rect is None:
                        continue
                    x1,y1,x2,y2 = rect
                    tpl = self.frame[y1:y2, x1:x2]
                    ts = time.time_ns()
                    path = os.path.join("templates", f"tpl_{ts}.png")
                    if cv2.imwrite(path, tpl):
                        print(f"Saved template: {path}")
                        self._push_action("wait_image", template=path, threshold=0.92, timeout=10.0)
                    else:
                        print(f"[wait_image] imwrite failed for {path}")

                elif key == ord('c'):
                    if self.actions:
                        removed = self.actions.pop()
                        print(f"- undo {removed.type}")

                elif key == ord('s'):
                    self._save()
                    break

                elif key == ord('q'):
                    print("Quit without saving")
                    break

                elif key == ord('m'):
                    self._push_action("match3_once", grid=_default_grid_path(), mode="swipe")

                elif key == ord('a'):
                    self._push_action("match3_auto", grid=_default_grid_path(), mode="swipe",
                                    max_moves=120, delay=0.35)
                    print("[rec] inserted: match3_auto (max_moves=120, delay=0.35)")

                elif key == ord('f'):
                    self._push_action("match3_autoplay_start", grid=_default_grid_path(),
                                    mode="swipe", delay=0.35)
                    print("[rec] inserted: match3_autoplay_start")

                elif key == ord('g'):
                    self._push_action("match3_autoplay_stop")
                    print("[rec] inserted: match3_autoplay_stop")

                elif key == ord('u'):
                    if self.frame is None:
                        continue
                    rect = self._select_rect()
                    cv2.setMouseCallback(self.window, self._on_mouse)
                    if rect is None:
                        continue
                    x1,y1,x2,y2 = rect
                    tpl = self.frame[y1:y2, x1:x2]
                    ts = time.time_ns()
                    path = os.path.join("templates", f"tpl_{ts}.png")
                    if cv2.imwrite(path, tpl):
                        print(f"Saved template: {path}")
                        self._push_action("match3_autoplay_until",
                                        grid=_default_grid_path(), mode="swipe", delay=0.35,
                                        template=path, threshold=0.92, timeout=25.0, disappear=False)
                    else:
                        print(f"[wait_image] imwrite failed for {path}")

        finally:
            stream.stop()
            # Закрываем только окно предпросмотра, чтобы не уронить Tkinter GUI
            try:
                cv2.destroyWindow(self.window)  # self.window = "ADB Recorder" (или как у тебя названо)
            except Exception:
                pass
            time.sleep(0.05)  # чуть-чуть подождём, чтобы HighGUI корректно отпустил окно



class Player:
    def __init__(self, adb: ADB):
        self.adb = adb
        self.res = self.adb.resolution()
        # --- autoplay state ---
        self._ap_thread = None
        self._ap_stop = threading.Event()

    def play(self, scenario_path: str):
        with open(scenario_path, "r", encoding="utf-8") as f:
            steps: List[Dict[str, Any]] = yaml.safe_load(f) or []

        for i, raw_step in enumerate(steps, 1):
            step = dict(raw_step)  # не модифицируем оригинал
            typ = step.get("type")
            retries = int(step.get("retries", 0))
            retry_delay = float(step.get("retry_delay", 0.3))
            wait_before = float(step.get("wait_before", 0.0))
            wait_after = float(step.get("wait_after", 0.0))

            name = step.get("name")
            label = f"{typ} — {name}" if name else typ
            print(f"[{i}/{len(steps)}] {label}")

            if wait_before > 0:
                time.sleep(wait_before)

            ok = False
            for attempt in range(retries + 1):
                try:
                    ok = self._do_step_once(typ, step)
                except Exception as e:
                    print(f"[play] exception at step {i} ({typ}): {e}")
                    self._dump_failure(i, step, reason=str(e))
                    ok = False

                if not ok:
                    if attempt < retries:
                        time.sleep(retry_delay)
                        continue
                    else:
                        break

                # пост-условие (если есть expect/expect_image)
                expect = step.get("expect") or step.get("expect_image")
                if expect:
                    thr = float(step.get("expect_threshold", 0.90))
                    timeout = float(step.get("expect_timeout", 8.0))
                    disappear = bool(step.get("expect_disappear", False))
                    ok = self._wait_image_any(expect, thr=thr, timeout=timeout, disappear=disappear)
                    if not ok and attempt < retries:
                        print("[expect] not met, retrying step...")
                        time.sleep(retry_delay)
                        continue
                break  # шаг выполнен

            if not ok:
                print(f"[play] step failed, stopping: {typ}")
                self._dump_failure(i, step, reason="step returned False")
                self._stop_autoplay()  # на всякий случай гасим фон
                break

            if wait_after > 0:
                time.sleep(wait_after)

        # гарантированно останавливаем фон по завершению сценария
        self._stop_autoplay()

    # ---------- вспомогательные дампы при падении ----------
    def _dump_failure(self, i, step, reason=""):
        try:
            os.makedirs("artifacts", exist_ok=True)
        except Exception:
            pass

        ts = time.strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join("artifacts", f"fail_{i:03d}_{step.get('type','unknown')}_{ts}.png")
        meta_path = os.path.join("artifacts", f"fail_{i:03d}_{ts}.yaml")

        try:
            frame = self.adb.screencap()
            if frame is not None:
                cv2.imwrite(img_path, frame)
                print(f"[fail] screenshot saved: {img_path}")
        except Exception as e:
            print(f"[fail] could not take screenshot: {e}")

        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                yaml.safe_dump({"index": i, "step": step, "reason": reason},
                               f, allow_unicode=True, sort_keys=False)
            print(f"[fail] meta saved: {meta_path}")
        except Exception as e:
            print(f"[fail] could not save meta: {e}")

    # ---------- исполнение одного шага ----------
    def _do_step_once(self, typ: str, step: Dict[str, Any]) -> bool:
        if typ == "tap":
            x, y = norm_to_px(tuple(step["at"]), self.res)
            self.adb.tap(x, y)
            return True

        elif typ == "swipe":
            x1, y1 = norm_to_px(tuple(step["from"]), self.res)
            x2, y2 = norm_to_px(tuple(step["to"]), self.res)
            ms = int(step.get("ms", 180))
            self.adb.swipe(x1, y1, x2, y2, ms)
            return True

        elif typ == "wait":
            time.sleep(float(step.get("seconds", 1.0)))
            return True

        elif typ == "wait_image":
            tpl = step["template"]
            thr = float(step.get("threshold", 0.92))
            timeout = float(step.get("timeout", 10.0))
            return self._wait_image_any(tpl, thr=thr, timeout=timeout, disappear=False)

        elif typ == "wait_disappear":
            tpl = step["template"]
            thr = float(step.get("threshold", 0.92))
            timeout = float(step.get("timeout", 10.0))
            return self._wait_image_any(tpl, thr=thr, timeout=timeout, disappear=True)

        elif typ == "tap_image":
            tpl = step["template"]
            thr = float(step.get("threshold", 0.90))
            center = self._find_image_center(tpl, thr=thr)
            if center is None:
                return False
            dx = int(step.get("offset_x", 0))
            dy = int(step.get("offset_y", 0))
            self.adb.tap(center[0] + dx, center[1] + dy)
            return True

        elif typ == "wait_stable":
            roi = step.get("roi")  # [x1,y1,x2,y2] абсолют или нормализованные 0..1
            eps = float(step.get("eps", 3.0))
            stable_for = float(step.get("stable_for", 0.6))
            timeout = float(step.get("timeout", 8.0))
            return self._wait_stable(roi=roi, eps=eps, stable_for=stable_for, timeout=timeout)

        elif typ == "match3_once":
            grid = step.get("grid", "grid.yaml")
            mode = step.get("mode", "swipe")
            ok = match3_once(self.adb, grid_yaml=grid, action_mode=mode)
            return ok

        # одиночный автоплей (блокирующий, X ходов)
        elif typ == "match3_auto":
            grid = step.get("grid", "grid.yaml")
            mode = step.get("mode", "swipe")
            max_moves = int(step.get("max_moves", 100))
            delay = float(step.get("delay", 0.30))
            match3_autoplay(self.adb, grid_yaml=grid, mode=mode, max_moves=max_moves, delay=delay)
            return True

        # фоновый автоплей: старт/стоп
        elif typ == "match3_autoplay_start":
            grid = step.get("grid", "grid.yaml")
            mode = step.get("mode", "swipe")
            delay = float(step.get("delay", 0.35))
            return self._start_autoplay(grid, mode=mode, delay=delay)

        elif typ == "match3_autoplay_stop":
            return self._stop_autoplay()


        elif typ == "match3_autoplay_until":
            grid = step.get("grid", _default_grid_path())
            mode = step.get("mode", "swipe")
            delay = float(step.get("delay", 0.35))

            # Картинка (опционально)
            tpl_path = step.get("template")
            thr = float(step.get("threshold", 0.92))
            disappear = bool(step.get("disappear", False))

            # Таймаут (страховочный). По умолчанию НЕ валим сценарий.
            timeout = float(step.get("timeout", 600.0))
            on_timeout = step.get("on_timeout", "continue")  # "continue" | "fail"

            # Сколько подряд «нет хода» считаем концом поля/каскада
            no_move_patience = int(step.get("no_move_patience", 8))

            # Детектор «поле видно»
            board_min_std   = float(step.get("board_min_std",   7.0))
            board_min_edges = float(step.get("board_min_edges", 0.06))  # 6% пикселей — грани
            board_patience  = int(step.get("board_patience", 3))        # сколько подряд кадров «не видно»

            # Шаблон (если задан)
            tpl = None
            if tpl_path:
                tpl = cv2.imread(tpl_path, cv2.IMREAD_COLOR)
                if tpl is None:
                    print(f"[match3_until] template not found: {tpl_path}")

            board_rect = self._board_rect_from_grid(grid, margin=6)

            def _tpl_condition_met() -> bool:
                if tpl is None:
                    return False
                frame = self.adb.screencap()
                score, _ = self._match_template(frame, tpl, thr)
                present = (score >= thr)
                return (present and not disappear) or ((not present) and disappear)

            t0 = time.time()
            no_move = 0
            board_not_vis = 0

            while True:
                # 0) Глобальный таймаут
                if time.time() - t0 > timeout:
                    print("[match3_until] timeout")
                    return (on_timeout != "fail")  # по умолчанию не валим сценарий

                # 1) Если задан шаблон — он имеет приоритет для остановки
                if _tpl_condition_met():
                    print("[match3_until] template condition met — stop")
                    return True

                # 2) Проверяем «доска видна»
                frame = self.adb.screencap()
                if not self._board_visible_on_frame(frame, board_rect, min_std=board_min_std, min_edges=board_min_edges):
                    board_not_vis += 1
                    if board_not_vis >= board_patience:
                        print("[match3_until] board gone — stop")
                        return True
                else:
                    board_not_vis = 0

                # 3) Делаем один ход
                moved = match3_once(self.adb, grid_yaml=grid, action_mode=mode)
                if moved:
                    no_move = 0
                else:
                    no_move += 1
                    if no_move >= no_move_patience:
                        print("[match3_until] no moves — stop")
                        return True

                time.sleep(delay if moved else 0.30)




        else:
            print(f"[warn] unknown step: {typ}")
            return True  # не валим сценарий на незнакомом шаге

    # ---------- sync helpers ----------
    def _wait_image(self, step: Dict[str,Any]):
        template_path = step["template"]
        thr = float(step.get("threshold", 0.92))
        timeout = float(step.get("timeout", 10.0))
        tpl = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if tpl is None:
            print(f"[wait_image] template not found: {template_path}")
            return
        th, tw = tpl.shape[:2]
        t0 = time.time()
        while True:
            frame = self.adb.screencap()
            res = cv2.matchTemplate(frame, tpl, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, maxloc = cv2.minMaxLoc(res)
            if maxv >= thr:
                cx = maxloc[0] + tw//2
                cy = maxloc[1] + th//2
                print(f"[wait_image] found at ({cx},{cy}) score={maxv:.3f}")
                break
            if time.time() - t0 > timeout:
                print("[wait_image] timeout")
                break
            time.sleep(0.2)

    def _match_template(self, frame, tpl, thr):
        res = cv2.matchTemplate(frame, tpl, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        return maxv, maxloc

    def _wait_image_any(self, template_path: str, thr: float = 0.90, timeout: float = 8.0, disappear: bool = False) -> bool:
        tpl = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if tpl is None:
            print(f"[wait_image] template not found: {template_path}")
            return False
        t0 = time.time()
        while True:
            frame = self.adb.screencap()
            score, _ = self._match_template(frame, tpl, thr)
            ok = (score >= thr)
            if (ok and not disappear) or ((not ok) and disappear):
                return True
            if time.time() - t0 > timeout:
                return False
            time.sleep(0.2)

    def _find_image_center(self, template_path: str, thr: float = 0.90):
        tpl = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if tpl is None:
            print(f"[tap_image] template not found: {template_path}")
            return None
        th, tw = tpl.shape[:2]
        frame = self.adb.screencap()
        score, loc = self._match_template(frame, tpl, thr)
        if score < thr:
            return None
        cx = loc[0] + tw // 2
        cy = loc[1] + th // 2
        return (cx, cy)

    def _parse_roi(self, roi):
        if roi is None:
            return None
        x1, y1, x2, y2 = roi
        if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
            x1, y1 = norm_to_px((x1, y1), self.res)
            x2, y2 = norm_to_px((x2, y2), self.res)
        return (max(0, int(x1)), max(0, int(y1)), min(self.res[0]-1, int(x2)), min(self.res[1]-1, int(y2)))

    def _wait_stable(self, roi=None, eps: float = 3.0, stable_for: float = 0.6, timeout: float = 8.0) -> bool:
        rect = self._parse_roi(roi) if roi is not None else None
        prev = None
        t_stable = None
        t0 = time.time()
        while True:
            frame = self.adb.screencap()
            crop = frame if rect is None else frame[rect[1]:rect[3], rect[0]:rect[2]]
            if prev is not None:
                diff = cv2.absdiff(crop, prev)
                mean = float(np.mean(diff))
                if mean <= eps:
                    if t_stable is None:
                        t_stable = time.time()
                    elif time.time() - t_stable >= stable_for:
                        return True
                else:
                    t_stable = None
            prev = crop
            if time.time() - t0 > timeout:
                return False
            time.sleep(0.1)

    def _board_rect_from_grid(self, grid_yaml: str, margin: int = 6):
        g = load_grid_yaml(grid_yaml)
        x1, y1 = g.top_left
        x2, y2 = g.bot_right
        # нормализуем так, чтобы x1<x2, y1<y2 и добавим небольшой отступ
        x1, x2 = sorted([int(x1), int(x2)])
        y1, y2 = sorted([int(y1), int(y2)])
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(self.res[0] - 1, x2 + margin)
        y2 = min(self.res[1] - 1, y2 + margin)
        return (x1, y1, x2, y2)

    def _board_visible_on_frame(self, frame, rect, min_std=7.0, min_edges=0.06) -> bool:
        x1, y1, x2, y2 = rect
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        std  = float(np.std(gray))

        # плотность граней
        edges = cv2.Canny(gray, 60, 120)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)

        # доска «видна», если есть и текстура, и достаточное количество граней
        return (std >= min_std) and (edge_density >= min_edges)



    # ---------- autoplay ----------
    def _autoplay_loop(self, grid, mode, delay):
        print(f"[autoplay] started (grid={grid}, mode={mode}, delay={delay})")
        try:
            while not self._ap_stop.is_set():
                ok = match3_once(self.adb, grid_yaml=grid, action_mode=mode)
                time.sleep(delay if ok else 0.30)
        finally:
            print("[autoplay] stopped")

    def _start_autoplay(self, grid, mode="swipe", delay=0.35):
        if self._ap_thread and self._ap_thread.is_alive():
            print("[autoplay] already running")
            return True
        self._ap_stop.clear()
        self._ap_thread = threading.Thread(
            target=self._autoplay_loop, args=(grid, mode, delay), daemon=True
        )
        self._ap_thread.start()
        return True

    def _stop_autoplay(self):
        if not self._ap_thread:
            return True
        self._ap_stop.set()
        self._ap_thread.join(timeout=2.0)
        self._ap_thread = None
        return True




# ---------------------------- Match-3 ----------------------------

def fit_to_screen(img: np.ndarray, max_w: int = 1080, max_h: int = 720) -> Tuple[np.ndarray, float]:
    h,w = img.shape[:2]
    scale = min(max_w/w, max_h/h, 1.0)
    if scale < 1.0:
        out = cv2.resize(img, (int(w*scale), int(h*scale)))
        return out, scale
    return img, 1.0

@dataclass
class Grid:
    rows: int
    cols: int
    top_left: Tuple[int,int]
    bot_right: Tuple[int,int]

    @property
    def cell_centers(self) -> List[Tuple[int,int]]:
        x1,y1 = self.top_left
        x2,y2 = self.bot_right
        rows, cols = self.rows, self.cols
        xs = np.linspace(x1, x2, cols)
        ys = np.linspace(y1, y2, rows)
        centers = []
        for r in range(rows):
            for c in range(cols):
                centers.append((int(round(xs[c])), int(round(ys[r]))))
        return centers

    def center_of(self, r: int, c: int) -> Tuple[int,int]:
        return self.cell_centers[r*self.cols + c]


def save_grid_yaml(grid: Grid, path: str):
    data = {
        "rows": grid.rows,
        "cols": grid.cols,
        "top_left": list(grid.top_left),
        "bot_right": list(grid.bot_right),
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_grid_yaml(path: str) -> Grid:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return Grid(rows=int(d["rows"]), cols=int(d["cols"]), top_left=tuple(d["top_left"]), bot_right=tuple(d["bot_right"]))


def calibrate_grid(adb: ADB, rows: int, cols: int, out_path: str):
    frame = adb.screencap()
    win = "Calibrate Grid: click TL cell center, then BR cell center"
    pts = []

    def cb(ev, x, y, flags, ud):
        nonlocal pts, scale
        if ev == cv2.EVENT_LBUTTONDOWN:
            fx, fy = int(x/scale), int(y/scale)
            pts.append((fx, fy))

    disp, scale = fit_to_screen(frame)
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, cb)
    while True:
        draw = disp.copy()
        for p in pts:
            cv2.circle(draw, (int(p[0]*scale), int(p[1]*scale)), 6, (0,255,0), -1)
        cv2.imshow(win, draw)
        key = cv2.waitKey(1) & 0xFF
        if len(pts) >= 2:
            break
        if key == 27:  # ESC
            cv2.destroyWindow(win)
            print("Cancelled")
            return
    cv2.destroyWindow(win)
    tl, br = pts[0], pts[1]
    grid = Grid(rows, cols, tl, br)
    save_grid_yaml(grid, out_path)
    print(f"Saved grid: {out_path}")


def avg_hsv(img: np.ndarray, rect: Tuple[int,int,int,int]) -> Tuple[float,float,float]:
    x1,y1,x2,y2 = rect
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return (0.0,0.0,0.0)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return tuple(np.mean(hsv.reshape(-1,3), axis=0).tolist())  # H,S,V


def board_colors(frame: np.ndarray, grid: Grid) -> np.ndarray:
    rows, cols = grid.rows, grid.cols
    x1,y1 = grid.top_left
    x2,y2 = grid.bot_right
    cw = (x2 - x1) / max(cols-1,1)
    ch = (y2 - y1) / max(rows-1,1)
    colors = np.zeros((rows, cols, 3), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            cx, cy = grid.center_of(r,c)
            # small square around center (1/3 of cell step)
            side = max(3, int(min(cw, ch) * 0.2))
            rw = side
            rh = side
            rect = (max(0,cx-rw), max(0,cy-rh), min(frame.shape[1], cx+rw), min(frame.shape[0], cy+rh))
            colors[r,c] = avg_hsv(frame, rect)
    return colors


def hsv_close(a: np.ndarray, b: np.ndarray, tol_h=8, tol_s=35, tol_v=35) -> bool:
    dh = min(abs(a[0]-b[0]), 180 - abs(a[0]-b[0]))
    ds = abs(a[1]-b[1])
    dv = abs(a[2]-b[2])
    return (dh <= tol_h) and (ds <= tol_s) and (dv <= tol_v)



def simulate_swap_and_check(colors: np.ndarray, r1:int,c1:int, r2:int,c2:int) -> bool:
    tmp = colors.copy()
    tmp[r1,c1], tmp[r2,c2] = tmp[r2,c2].copy(), tmp[r1,c1].copy()
    rows, cols = tmp.shape[:2]
    # check horizontal lines
    for r in range(rows):
        run = 1
        for c in range(1, cols):
            if hsv_close(tmp[r,c], tmp[r,c-1]):
                run += 1
                if run >= 3:
                    return True
            else:
                run = 1
    # check vertical lines
    for c in range(cols):
        run = 1
        for r in range(1, rows):
            if hsv_close(tmp[r,c], tmp[r-1,c]):
                run += 1
                if run >= 3:
                    return True
            else:
                run = 1
    return False


def find_any_valid_move(colors: np.ndarray) -> Optional[Tuple[Tuple[int,int], Tuple[int,int]]]:
    rows, cols = colors.shape[:2]
    for r in range(rows):
        for c in range(cols):
            if c+1 < cols:  # right
                if simulate_swap_and_check(colors, r,c, r,c+1):
                    return (r,c), (r,c+1)
            if r+1 < rows:  # down
                if simulate_swap_and_check(colors, r,c, r+1,c):
                    return (r,c), (r+1,c)
    return None


def match3_once(adb: ADB, grid_yaml: str, action_mode: str = "swipe") -> bool:
    grid = load_grid_yaml(grid_yaml)
    frame = adb.screencap()
    cols = board_colors(frame, grid)
    move = find_any_valid_move(cols)
    if not move:
        return False
    (r1,c1),(r2,c2) = move
    x1,y1 = grid.center_of(r1,c1)
    x2,y2 = grid.center_of(r2,c2)
    print(f"[match3] move {(r1,c1)} -> {(r2,c2)} ({x1},{y1}) -> ({x2},{y2})")
    if action_mode == "swipe":
        adb.swipe(x1,y1,x2,y2,150)
    else:
        adb.tap(x1,y1)
        time.sleep(0.05)
        adb.tap(x2,y2)
    # small wait for animation
    time.sleep(0.5)
    return True

def match3_autoplay(adb: ADB, grid_yaml: str, mode: str = "swipe",
                    max_moves: int = 200, delay: float = 0.30,
                    stop_event: Optional[threading.Event] = None) -> int:
    """
    Крутит последовательные ходы match-3, пока есть валидные или пока
    не исчерпан лимит ходов. Можно прервать через stop_event.set().
    Возвращает количество выполненных ходов.
    """
    moves = 0
    for _ in range(max_moves):
        if stop_event is not None and stop_event.is_set():
            print("[autoplay] interrupted by user")
            break
        ok = match3_once(adb, grid_yaml=grid_yaml, action_mode=mode)
        if not ok:
            print("[autoplay] no move found — stopping")
            break
        moves += 1
        if delay > 0:
            time.sleep(delay)
    return moves



# ---------------------------- CLI ----------------------------

def main():
    ap = argparse.ArgumentParser(description="ADB Scenario Bot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("record", help="Record scenario")
    p_rec.add_argument("--out", required=True, help="Output YAML path")
    p_rec.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL"))
    p_rec.add_argument("--live", action="store_true", help="also perform on device while recording")


    p_play = sub.add_parser("play", help="Play scenario")
    p_play.add_argument("--in", dest="inp", required=True, help="Input YAML path")
    p_play.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL"))

    p_cal = sub.add_parser("calibrate", help="Calibrate match3 grid")
    p_cal.add_argument("--rows", type=int, required=True)
    p_cal.add_argument("--cols", type=int, required=True)
    p_cal.add_argument("--out", required=True)
    p_cal.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL"))

    p_m3 = sub.add_parser("match3", help="Perform one match3 move")
    p_m3.add_argument("--grid", required=True)
    p_m3.add_argument("--mode", choices=["swipe","tap"], default="swipe")
    p_m3.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL"))

    p_auto = sub.add_parser("autoplay", help="Loop match-3 moves")
    p_auto.add_argument("--grid", required=True)
    p_auto.add_argument("--mode", choices=["swipe","tap"], default="swipe")
    p_auto.add_argument("--max-moves", type=int, default=200)
    p_auto.add_argument("--delay", type=float, default=0.30)
    p_auto.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL"))

    p_mirr = sub.add_parser("mirror", help="Open live mirror window (HighGUI)")
    p_mirr.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL"))
    p_mirr.add_argument("--scale", type=float, default=1.0)

    args = ap.parse_args()

    adb = ADB(serial=args.serial)

    if args.cmd == "record":
        Recorder(adb, out_path=args.out, mirror_live=args.live).run()

    elif args.cmd == "play":
        Player(adb).play(args.inp)
    elif args.cmd == "calibrate":
        calibrate_grid(adb, rows=args.rows, cols=args.cols, out_path=args.out)
    elif args.cmd == "match3":
        match3_once(adb, grid_yaml=args.grid, action_mode=args.mode)
    elif args.cmd == "mirror":
        run_mirror(ADB(serial=args.serial), scale=args.scale)

    elif args.cmd == "autoplay":
        n = match3_autoplay(adb, grid_yaml=args.grid, mode=args.mode,
                            max_moves=args.max_moves, delay=args.delay)
        print(f"[autoplay] moves executed: {n}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print("ADB error:", e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted by user")
        sys.exit(130)
