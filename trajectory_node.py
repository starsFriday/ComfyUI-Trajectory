import json
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional
import math

import numpy as np
import torch
from PIL import Image, ImageDraw

Color = Tuple[int, int, int, int]
Point = Tuple[float, float]


@dataclass
class TrajectoryPayload:
    strokes: List[List[Point]]
    reference_image: Optional[str] = None
    reference_size: Optional[Tuple[int, int]] = None

    @classmethod
    def from_json(cls, raw: str) -> "TrajectoryPayload":
        if not raw or not raw.strip():
            raise ValueError("Trajectory data is empty. Hold Shift and draw a path first.")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Trajectory data is not valid JSON: {exc}") from exc

        raw_strokes = payload.get("strokes") if isinstance(payload, dict) else None
        if not raw_strokes:
            raise ValueError("Trajectory data does not contain any strokes.")

        strokes: List[List[Point]] = []
        for stroke in raw_strokes:
            pts = stroke.get("points") if isinstance(stroke, dict) else stroke
            if not pts:
                continue
            processed: List[Point] = []
            for point in pts:
                x = y = None
                if isinstance(point, dict):
                    x = point.get("x")
                    y = point.get("y")
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    x, y = point[0], point[1]
                if x is None or y is None:
                    continue
                processed.append((float(x), float(y)))
            if processed:
                strokes.append(processed)

        reference_image = None
        reference_size = None
        if isinstance(payload, dict):
            reference_image = payload.get("reference_image")
            ref_size = payload.get("reference_size")
            if isinstance(ref_size, dict):
                width = ref_size.get("width")
                height = ref_size.get("height")
                if isinstance(width, (int, float)) and isinstance(height, (int, float)):
                    reference_size = (int(width), int(height))

        if not strokes:
            raise ValueError("No valid coordinates detected inside the trajectory data.")
        return cls(strokes=strokes, reference_image=reference_image, reference_size=reference_size)


class TrajectoryDrawer:
    CATEGORY = "🧭 Trajectory"
    RETURN_NAMES = ("frames", "info", "tracks_json")
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    FUNCTION = "draw"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "path_json": ("STRING", {"multiline": True, "default": "", "forceInput": False}),
                "stroke_width": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 200.0, "step": 0.5}),
                "frames_per_segment": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1}),
                "line_color": ("STRING", {"default": "#00f0ff"}),
                "tail_alpha": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                "background_dim": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "trail_decay": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "line_style": (["solid", "neon"],),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120, "step": 1}),
                "duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 120.0, "step": 0.1}),
                "show_pointer": (["disable", "enable"],),
            }
        }

    @staticmethod
    def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
        image_np = image_tensor.detach().cpu().numpy()
        image_np = np.clip(image_np, 0.0, 1.0)
        image_uint8 = (image_np * 255.0).astype(np.uint8)
        return Image.fromarray(image_uint8)

    @staticmethod
    def _pil_to_tensor(frames: Sequence[Image.Image]) -> torch.Tensor:
        stacked = np.stack([(np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0) for frame in frames])
        return torch.from_numpy(stacked)

    @staticmethod
    def _parse_color(value: str, alpha: float) -> Color:
        value = (value or "#ffffff").strip()
        if value.startswith("#"):
            value = value.lstrip("#")
            if len(value) == 3:
                value = "".join(ch * 2 for ch in value)
            if len(value) != 6:
                raise ValueError("Hex colors must be 3 or 6 characters long.")
            r, g, b = tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
        else:
            try:
                parts = [int(p.strip()) for p in value.split(",")]
                if len(parts) != 3:
                    raise ValueError
                r, g, b = parts
            except Exception as exc:
                raise ValueError("Unsupported color format. Use hex (#00ffaa) or r,g,b") from exc
        alpha_channel = int(np.clip(alpha, 0.0, 1.0) * 255)
        return r, g, b, alpha_channel

    @staticmethod
    def _normalize_points(points: Iterable[Point]) -> List[Point]:
        normalized = []
        for x, y in points:
            normalized.append((float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))))
        return normalized

    @staticmethod
    def _sample_points(strokes: List[List[Point]], steps_per_segment: int) -> List[List[Point]]:
        sampled: List[List[Point]] = []
        for stroke in strokes:
            norm = TrajectoryDrawer._normalize_points(stroke)
            if len(norm) < 2:
                continue
            resampled: List[Point] = []
            for idx in range(len(norm) - 1):
                start = norm[idx]
                end = norm[idx + 1]
                resampled.append(start)
                for step in range(1, steps_per_segment + 1):
                    t = step / (steps_per_segment + 1)
                    interp = (start[0] * (1 - t) + end[0] * t, start[1] * (1 - t) + end[1] * t)
                    resampled.append(interp)
            resampled.append(norm[-1])
            sampled.append(resampled)
        return sampled

    def _build_frames(
        self,
        base: Image.Image,
        sampled_strokes: Sequence[Sequence[Point]],
        color: Color,
        stroke_width: float,
        trail_decay: float,
        line_style: str,
        fps: int,
        duration_seconds: float,
        show_pointer: bool,
    ) -> Tuple[List[Image.Image], List[List[dict]]]:
        valid_strokes = [tuple(stroke) for stroke in sampled_strokes if len(stroke) >= 2]
        if not valid_strokes:
            raise ValueError("Need at least two points to produce an animation.")

        w, h = base.size
        frames: List[Image.Image] = []

        pixel_strokes: List[List[Tuple[float, float]]] = [
            [(point[0] * (w - 1), point[1] * (h - 1)) for point in stroke]
            for stroke in valid_strokes
        ]

        if duration_seconds > 0.0:
            frame_count = max(1, int(round(max(duration_seconds, 0.01) * max(fps, 1))))
        else:
            frame_count = max(1, max(len(stroke) - 1 for stroke in pixel_strokes))
        resample_length = frame_count + 1

        resampled_strokes = [
            self._resample_points(stroke, resample_length) for stroke in pixel_strokes
        ]

        history_frames = max(
            2,
            int(
                round(max(0.0, 1.0 - float(np.clip(trail_decay, 0.0, 1.0))) * max(fps, 1) * 2.0)
            )
            + 2,
        )
        track_records = [
            [{"x": int(round(pt[0])), "y": int(round(pt[1]))} for pt in stroke_points]
            for stroke_points in resampled_strokes
        ]

        circle_radius = max(2.0, stroke_width * 1.25)
        circle_alpha = int(np.clip(color[3], 0, 255))
        pointer_directions = [(1.0, 0.0)] * len(resampled_strokes)
        pointer_scale = max(6.0, stroke_width * 1.75)

        for idx in range(resample_length):
            frame = base.copy()
            circle_overlay = None
            line_overlay = None
            pointer_overlay = None
            pointer_draw = None

            for stroke_idx, stroke_points in enumerate(resampled_strokes):
                current = stroke_points[idx]

                if circle_alpha > 0:
                    if circle_overlay is None:
                        circle_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                    draw_circle = ImageDraw.Draw(circle_overlay, "RGBA")
                    draw_circle.ellipse(
                        (
                            current[0] - circle_radius,
                            current[1] - circle_radius,
                            current[0] + circle_radius,
                            current[1] + circle_radius,
                        ),
                        fill=color,
                    )

                tail_coords = stroke_points[max(0, idx - history_frames) : idx + 1]
                if len(tail_coords) > 1:
                    pointer_directions[stroke_idx] = self._normalized_direction(
                        tail_coords[-2],
                        tail_coords[-1],
                        pointer_directions[stroke_idx],
                    )
                    if line_overlay is None:
                        line_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                    self._draw_gradient_polyline_on_overlay(
                        line_overlay,
                        int(max(1, round(stroke_width))),
                        tail_coords,
                        color[:3],
                        line_style,
                    )

                if show_pointer:
                    if pointer_overlay is None:
                        pointer_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                        pointer_draw = ImageDraw.Draw(pointer_overlay, "RGBA")
                    self._draw_pointer(
                        pointer_draw,
                        current,
                        pointer_directions[stroke_idx],
                        pointer_scale,
                    )

            if circle_overlay is not None:
                frame = Image.alpha_composite(frame, circle_overlay)
            if line_overlay is not None:
                frame = Image.alpha_composite(frame, line_overlay)
            if pointer_overlay is not None:
                frame = Image.alpha_composite(frame, pointer_overlay)

            frames.append(frame.convert("RGB"))
        return frames, track_records

    @staticmethod
    def _resample_points(points: Sequence[Tuple[float, float]], desired: int) -> List[Tuple[float, float]]:
        if desired <= 2 or len(points) <= 2:
            return list(points)
        distances = [0.0]
        total = 0.0
        for idx in range(1, len(points)):
            start = points[idx - 1]
            end = points[idx]
            seg_len = math.hypot(end[0] - start[0], end[1] - start[1])
            total += seg_len
            distances.append(total)
        if total == 0:
            return list(points)

        resampled: List[Tuple[float, float]] = []
        seg_idx = 0
        for frame in range(desired):
            target = (total * frame) / (desired - 1)
            while seg_idx < len(distances) - 1 and distances[seg_idx + 1] < target:
                seg_idx += 1
            start = points[seg_idx]
            end = points[min(seg_idx + 1, len(points) - 1)]
            seg_start = distances[seg_idx]
            seg_end = distances[min(seg_idx + 1, len(distances) - 1)]
            if seg_end - seg_start == 0:
                t = 0.0
            else:
                t = (target - seg_start) / (seg_end - seg_start)
            interp = (start[0] * (1 - t) + end[0] * t, start[1] * (1 - t) + end[1] * t)
            resampled.append(interp)
        return resampled

    @staticmethod
    def _draw_gradient_polyline_on_overlay(
        img: Image.Image,
        line_width: int,
        coords: Sequence[Tuple[float, float]],
        color: Tuple[int, int, int],
        style: str,
    ) -> None:
        if len(coords) < 2:
            return
        draw = ImageDraw.Draw(img, "RGBA")
        steps = len(coords) - 1
        for idx in range(1, len(coords)):
            start = coords[idx - 1]
            end = coords[idx]
            strength = idx / steps
            if style == "neon":
                glow_width = max(line_width + 6, int(line_width * 2.2))
                glow_alpha = int(200 * strength)
                glow_color = (*color, glow_alpha)
                draw.line([start, end], fill=glow_color, width=glow_width, joint="curve")
            alpha = int(255 * strength)
            draw.line([start, end], fill=(*color, alpha), width=line_width, joint="curve")

    @staticmethod
    def _normalized_direction(
        start: Tuple[float, float],
        end: Tuple[float, float],
        fallback: Tuple[float, float],
    ) -> Tuple[float, float]:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        norm = math.hypot(dx, dy)
        if norm < 1e-5:
            return fallback
        return (dx / norm, dy / norm)

    @staticmethod
    def _draw_pointer(
        draw: ImageDraw.ImageDraw,
        position: Tuple[float, float],
        direction: Tuple[float, float],
        scale: float,
    ) -> None:
        px, py = position
        dx, dy = direction
        base_vec = (-dx, -dy)
        perp = (-dy, dx)
        tip = (px, py)
        base_center = (px + base_vec[0] * scale, py + base_vec[1] * scale)
        wing_spread = scale * 0.5
        base_left = (
            base_center[0] + perp[0] * wing_spread,
            base_center[1] + perp[1] * wing_spread,
        )
        base_right = (
            base_center[0] - perp[0] * wing_spread,
            base_center[1] - perp[1] * wing_spread,
        )
        fill_color = (255, 255, 255, 230)
        outline_color = (0, 0, 0, 200)
        draw.polygon([tip, base_left, base_right], fill=fill_color)
        draw.line([tip, base_left], fill=outline_color, width=1)
        draw.line([tip, base_right], fill=outline_color, width=1)
        draw.line([base_left, base_right], fill=outline_color, width=1)

    def draw(
        self,
        base_image: torch.Tensor,
        path_json: str,
        stroke_width: float,
        frames_per_segment: int,
        line_color: str,
        tail_alpha: float,
        background_dim: float,
        trail_decay: float,
        line_style: str,
        fps: int,
        duration_seconds: float,
        show_pointer: str,
    ):
        payload = TrajectoryPayload.from_json(path_json)
        sampled = self._sample_points(payload.strokes, frames_per_segment)

        first_frame = base_image[0].cpu()
        base = self._tensor_to_pil(first_frame)
        if background_dim > 0:
            base_arr = np.asarray(base).astype(np.float32) / 255.0
            base_arr *= 1.0 - np.clip(background_dim, 0.0, 1.0)
            base = Image.fromarray(np.clip(base_arr * 255.0, 0, 255).astype(np.uint8))
        base = base.convert("RGBA")

        color = self._parse_color(line_color, tail_alpha)
        frames, track_records = self._build_frames(
            base,
            sampled,
            color,
            stroke_width,
            trail_decay,
            line_style,
            fps,
            duration_seconds,
            show_pointer == "enable",
        )
        tensor_frames = self._pil_to_tensor(frames)
        track_json = json.dumps(track_records, ensure_ascii=False)
        track_points_preview = [
            point for stroke in track_records for point in stroke
        ]

        meta = {
            "frame_count": tensor_frames.shape[0],
            "stroke_points": sum(len(stroke) for stroke in payload.strokes),
            "resolution": list(base.size[::-1]),
            "fps": fps,
            "duration_seconds": duration_seconds if duration_seconds > 0 else None,
            "line_style": line_style,
            "trail_decay": trail_decay,
            "show_pointer": show_pointer == "enable",
            "track_points_preview": track_points_preview[:64],
            "tracks": len(track_records),
            "reference_size": payload.reference_size,
            "has_reference_image": payload.reference_image is not None,
        }
        return (tensor_frames, json.dumps(meta, ensure_ascii=False), track_json)


NODE_CLASS_MAPPINGS = {"TrajectoryDrawer": TrajectoryDrawer}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TrajectoryDrawer": "Trajectory ▸ Drawer",
}
