"""Note sequences and four explicit mirror views for osu!standard.

The transforms operate on map coordinates (including every slider anchor), not
on saved player results. They never rotate, rescale, clip or change timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .analyzer import ParsedMap, _sections, _pairs, normalize_mods

VERSION = 'note-sequence-v1'
VIEWS = ('original', 'x', 'y', 'xy')
PATH_POINTS = 8
FEATURES = ('x', 'y', 'dx_cs', 'dy_cs', 'log_dt', 'log_distance_cs', 'log_velocity_cs',
            'heading_x', 'heading_y', 'turn_sin', 'turn_cos', 'circle', 'slider', 'spinner',
            'log_duration', 'log_slider_length', 'log_repeats', 'beat_sin', 'beat_cos',
            'curve_linear', 'curve_bezier', 'curve_catmull', 'curve_perfect', 'log_anchors',
            'log_visible_notes', 'log_overlap',
            *(f'path_{i}_{axis}' for i in range(PATH_POINTS) for axis in ('x', 'y')))


@dataclass(frozen=True)
class NoteSequence:
    values: np.ndarray
    times_ms: np.ndarray
    end_times_ms: np.ndarray
    source_checksum: str
    mods: str
    view: str = 'original'


def mirror_map(parsed: ParsedMap, view: str) -> ParsedMap:
    if view not in VIEWS:
        raise ValueError('Allowed views: original, x, y, xy. Arbitrary rotations are disabled.')
    if parsed.metadata.get('augmentation', 'original') != 'original':
        raise ValueError('Views must be created from the original map, never chained.')
    if view == 'original':
        return parsed
    def point(x, y):
        return (512 - x if 'x' in view else x, 384 - y if 'y' in view else y)
    section = ''
    lines = []
    for line in parsed.content.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            section = stripped
        if section != '[HitObjects]' or not stripped or stripped.startswith(('[', '//')):
            lines.append(line)
            continue
        ending = '\r\n' if line.endswith('\r\n') else '\n' if line.endswith('\n') else ''
        fields = stripped.split(',')
        if len(fields) < 4:
            raise ValueError('Malformed hit object.')
        x, y = point(float(fields[0]), float(fields[1]))
        fields[0], fields[1] = format(x, '.12g'), format(y, '.12g')
        if int(fields[3]) & 2:
            curve = fields[5].split('|')
            for i in range(1, len(curve)):
                if ':' in curve[i]:
                    px, py = point(*map(float, curve[i].split(':')))
                    curve[i] = f'{px:.12g}:{py:.12g}'
            fields[5] = '|'.join(curve)
        lines.append(','.join(fields) + ending)
    content = ''.join(lines)
    metadata = {**parsed.metadata, 'augmentation': view, 'source_checksum': parsed.checksum}
    return ParsedMap(f'view-{view}-{parsed.map_id}', hashlib.md5(content.encode()).hexdigest(), content, metadata)


def mirror_cursor(points: np.ndarray, view: str) -> np.ndarray:
    if view not in VIEWS:
        raise ValueError('Unknown mirror view.')
    result = np.asarray(points).copy()
    if result.shape[-1] != 2:
        raise ValueError('Cursor points must have two coordinates.')
    if 'x' in view:
        result[..., 0] = 512 - result[..., 0]
    if 'y' in view:
        result[..., 1] = 384 - result[..., 1]
    return result


def mirror_sequence(sequence: NoteSequence, view: str) -> NoteSequence:
    if view not in VIEWS or sequence.view != 'original':
        raise ValueError('Choose one of four views of the original sequence.')
    if view == 'original':
        return sequence
    values = sequence.values.copy()
    for axis, position, displacement, heading in [('x', 'x', 'dx_cs', 'heading_x'), ('y', 'y', 'dy_cs', 'heading_y')]:
        if axis in view:
            values[:, FEATURES.index(position)] = 1 - values[:, FEATURES.index(position)]
            for name in (displacement, heading, *(f'path_{i}_{axis}' for i in range(PATH_POINTS))):
                values[:, FEATURES.index(name)] *= -1
    if view in ('x', 'y'):
        values[:, FEATURES.index('turn_sin')] *= -1
    return NoteSequence(values, sequence.times_ms.copy(), sequence.end_times_ms.copy(), sequence.source_checksum, sequence.mods, view)


def _timing(lines):
    events = []
    for line in lines:
        f = line.split(',')
        if len(f) >= 2:
            t, beat = float(f[0]), float(f[1])
            inherited = len(f) >= 7 and f[6].strip() == '0'
            events.append((t, beat, inherited))
    return sorted(events, key=lambda event: event[0])


def encode_map(parsed: ParsedMap, analysis: dict, view='original') -> NoteSequence:
    original_checksum = parsed.metadata.get('source_checksum', parsed.checksum)
    if view != 'original':
        parsed = mirror_map(parsed, view)
    sections = _sections(parsed.content)
    difficulty = _pairs(sections.get('Difficulty', []))
    multiplier = float(difficulty.get('SliderMultiplier', 1.4))
    mods, _ = normalize_mods(analysis.get('mods', 'NM'))
    clock = float(analysis.get('clock_rate', 1))
    if clock <= 0 or multiplier <= 0:
        raise ValueError('Invalid clock rate or slider multiplier.')
    events = _timing(sections.get('TimingPoints', []))
    objects = []
    event_index, beat, sv, beat_origin = 0, 500., 1., 0.
    for line in sections.get('HitObjects', []):
        fields = line.split(',')
        if len(fields) < 5:
            raise ValueError('Malformed note sequence.')
        x, y, t, kind = float(fields[0]), float(fields[1]), float(fields[2]), int(fields[3])
        while event_index < len(events) and events[event_index][0] <= t:
            et, length, inherited = events[event_index]
            if not inherited and length > 0:
                beat, sv, beat_origin = length, 1., et
            elif inherited and length < 0:
                sv = min(10., max(.1, -100 / length))
            event_index += 1
        curve, anchors, repeats, length, duration = '', [(x, y)], 0, 0., 0.
        if kind & 2:
            parts = fields[5].split('|')
            curve = parts[0]
            anchors += [tuple(map(float, p.split(':'))) for p in parts[1:] if ':' in p]
            repeats, length = int(fields[6]), float(fields[7])
            duration = length / (multiplier * 100 * sv) * beat * repeats
        elif kind & 8:
            duration = max(0., float(fields[5]) - t)
        if 'HR' in mods:
            y = 384 - y
            anchors = [(px, 384 - py) for px, py in anchors]
        objects.append((x, y, t / clock, duration / clock, kind, curve, anchors, repeats, length,
                        2 * math.pi * ((t - beat_origin) / beat % 1)))
    if not 2 <= len(objects) <= 100_000:
        raise ValueError('Sequence needs 2..100000 objects.')
    times = np.asarray([o[2] for o in objects], dtype=np.float64)
    if np.any(np.diff(times) < 0):
        raise ValueError('Hit objects must be in chronological order.')
    positions = np.asarray([(o[0], o[1]) for o in objects], dtype=np.float64)
    diameter = max(8., 2 * (54.4 - 4.48 * float(analysis.get('cs', 4)) ))
    preempt = float(analysis.get('preempt_ms', 1200))
    values = []
    previous_heading = np.zeros(2)
    for i, obj in enumerate(objects):
        x, y, time, duration, kind, curve, anchors, repeats, length, phase = obj
        delta = positions[i] - positions[i - 1] if i else np.zeros(2)
        distance = float(np.linalg.norm(delta))
        heading = delta / distance if distance > 1e-8 else np.zeros(2)
        dt = max(0, time - times[i - 1]) if i else 0.
        turn_sin = previous_heading[0] * heading[1] - previous_heading[1] * heading[0]
        turn_cos = float(previous_heading @ heading)
        stop = int(np.searchsorted(times, time + preempt, side='right'))
        visible = max(0, stop - i - 1)
        overlap = int(np.sum(np.linalg.norm(positions[i + 1:stop] - positions[i], axis=1) < diameter))
        # Raw control polygon landmarks preserve the curve type and all source
        # anchors in mirror_map; they are not claimed to be slider judgements.
        polygon = np.asarray(anchors, dtype=float)
        indices = np.linspace(0, max(0, len(polygon) - 1), PATH_POINTS)
        sampled = np.stack([np.interp(indices, np.arange(len(polygon)), polygon[:, j]) for j in range(2)], axis=1)
        sampled = ((sampled - positions[i]) / diameter).reshape(-1)
        signed = np.sign(delta) * np.log1p(np.abs(delta) / diameter)
        values.append([x / 512, y / 384, *signed, math.log1p(dt / 100), math.log1p(distance / diameter),
                       math.log1p(distance / diameter * 1000 / max(dt, 1)), *heading, turn_sin, turn_cos,
                       float(bool(kind & 1)), float(bool(kind & 2)), float(bool(kind & 8)),
                       math.log1p(duration / 100), math.log1p(length / diameter), math.log1p(repeats),
                       math.sin(phase), math.cos(phase), *(float(curve == c) for c in ('L', 'B', 'C', 'P')),
                       math.log1p(len(anchors)), math.log1p(visible), math.log1p(overlap), *sampled])
        previous_heading = heading
    array = np.asarray(values, dtype=np.float32)
    if array.shape[1] != len(FEATURES) or not np.isfinite(array).all():
        raise ValueError('Sequence contains invalid values.')
    end_times = times + np.asarray([o[3] for o in objects])
    return NoteSequence(array, times.astype(np.float32), end_times.astype(np.float32), original_checksum, mods, view)


def windows(sequence: NoteSequence, size=128, stride=64):
    if size < 2 or not 1 <= stride <= size:
        raise ValueError('Invalid window geometry.')
    values, length = sequence.values, len(sequence.values)
    starts = list(range(0, max(1, length - size + 1), stride))
    if length > size and starts[-1] != length - size:
        starts.append(length - size)
    result = np.zeros((len(starts), len(FEATURES), size), dtype=np.float32)
    masks = np.zeros((len(starts), size), dtype=bool)
    for i, start in enumerate(starts):
        chunk = values[start:start + size]
        result[i, :, :len(chunk)] = chunk.T
        masks[i, :len(chunk)] = True
    return result, masks, np.asarray(starts, dtype=np.int64)
