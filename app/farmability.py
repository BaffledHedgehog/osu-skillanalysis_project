"""Experimental pattern regularity descriptors, without a fixed accuracy bonus.

The deployed analyzer does not import this module. Descriptors are for a
separate, family-purged accuracy experiment, not a universal farm label.
"""
from bisect import bisect_right
from collections import Counter
import math

import numpy as np


FEATURES = ('rhythm_repeat_3', 'motif_repeat_3', 'motif_repeat_5',
            'rhythm_longest_run', 'rhythm_local_entropy_mean', 'rhythm_local_entropy_max',
            'spacing_entropy', 'angle_entropy', 'strain_variation',
            'strain_high_fraction', 'reading_density_squared_log',
            'reaction_pressure', 'reaction_pressure_squared', 'slider_length_variation')


def _entropy(items):
    counts = Counter(items)
    n = len(items)
    return -sum((c/n)*math.log2(c/n) for c in counts.values()) if n else 0.


def _repeat(segments, size):
    windows = [tuple(seq[i:i+size]) for seq in segments for i in range(len(seq)-size+1)]
    counts = Counter(windows)
    # Fraction belonging to repeated motifs, rather than repetitions caused by
    # a single token being counted several times inside an isolated pattern.
    return sum(n for n in counts.values() if n >= 2) / len(windows) if windows else 0.


def descriptors(content, analysis):
    section = ''; objects = []; timings = []; lengths = []
    for raw in content.splitlines():
        line = raw.strip()
        if line.startswith('['): section = line
        elif line and not line.startswith('//'):
            fields = line.split(',')
            try:
                if section == '[HitObjects]':
                    kind = int(fields[3])
                    if kind & 8: continue
                    objects.append((float(fields[2]),float(fields[0]),float(fields[1]),2 if kind&2 else 1))
                    if kind&2 and len(fields)>7: lengths.append(float(fields[7]))
                elif section == '[TimingPoints]' and float(fields[1])>0 and (len(fields)<7 or fields[6]=='1'):
                    timings.append((float(fields[0]),float(fields[1])))
            except (ValueError,IndexError): continue
    objects.sort(); timings.sort()
    offsets = [p[0] for p in timings]
    radius = max(1.,54.4-4.48*float(analysis.get('cs',4)))
    rhythm_segments = [[]]; motifs = [[]]; spacing = []; angles = []
    for i in range(1,len(objects)):
        a,b = objects[i-1],objects[i]
        gap = b[0]-a[0]
        clock = float(analysis.get('clock_rate',1))
        if not 20 <= gap/clock <= 2000:
            rhythm_segments.append([]); motifs.append([]); continue
        index = max(0,bisect_right(offsets,a[0])-1)
        beat = timings[index][1] if timings else 500.
        rhythm = round(math.log2(max(gap/beat,.01))*4)
        distance = math.hypot(b[1]-a[1],b[2]-a[2])/radius
        space = round(math.log2(1+distance)*2)
        angle = -1
        if i >= 2 and objects[i-1][0]-objects[i-2][0] <= 2000*clock:
            previous = objects[i-2]
            u = (a[1]-previous[1],a[2]-previous[2]); v = (b[1]-a[1],b[2]-a[2])
            norm = math.hypot(*u)*math.hypot(*v)
            if norm:
                cosine = max(-1.,min(1.,(u[0]*v[0]+u[1]*v[1])/norm))
                angle = round(math.acos(cosine)/math.pi*6)
                angles.append(angle)
        rhythm_segments[-1].append(rhythm)
        motifs[-1].append((rhythm,space,angle,b[3]))
        spacing.append(space)
    flat = [v for seq in rhythm_segments for v in seq]
    longest = 0
    for seq in rhythm_segments:
        run = 0; previous = None
        for item in seq:
            run = run+1 if item==previous else 1
            longest = max(longest,run); previous = item
    entropies = [_entropy(seq[i:i+16]) for seq in rhythm_segments
                 for i in range(0,max(0,len(seq)-7),8)]
    loads = np.asarray([w['load'] for w in analysis.get('windows',[]) if w['load']>0],dtype=float)
    mean = float(loads.mean()) if len(loads) else 0.
    preempt = max(1.,float(analysis.get('preempt_ms',600)))
    density = max(0.,float(analysis.get('reading_density',0)))
    values = (_repeat(rhythm_segments,3),_repeat(motifs,3),_repeat(motifs,5),
              longest/max(1,len(flat)),float(np.mean(entropies)) if entropies else 0.,
              max(entropies,default=0.),_entropy(spacing),_entropy(angles),
              float(loads.std()/mean) if mean else 0.,
              float(np.mean(loads>1.5*mean)) if mean else 0.,
              math.log1p(density)**2,1000/preempt,(1000/preempt)**2,
              float(np.std(lengths)/np.mean(lengths)) if lengths and np.mean(lengths)>0 else 0.)
    if not np.isfinite(values).all(): raise ValueError('Nonfinite pattern descriptor')
    return dict(zip(FEATURES,values))
