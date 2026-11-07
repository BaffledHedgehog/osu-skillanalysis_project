"""Map-derived reading descriptors; no fixed Hidden penalty or skill cutoff."""
from bisect import bisect_right
from collections import Counter
import math


def reading_features(content, preempt_ms, cs, clock_rate):
    section='';objects=[];timings=[]
    for line in content.splitlines():
        line=line.strip()
        if line.startswith('['):section=line
        elif line and not line.startswith('//'):
            fields=line.split(',')
            try:
                if section=='[HitObjects]' and not int(fields[3])&8:
                    objects.append((float(fields[2])/clock_rate,float(fields[0]),float(fields[1])))
                elif section=='[TimingPoints]' and float(fields[1])>0 and (len(fields)<7 or fields[6]=='1'):
                    timings.append((float(fields[0])/clock_rate,float(fields[1])/clock_rate))
            except (ValueError,IndexError):continue
    objects.sort();timings.sort()
    if not objects:return dict.fromkeys(('reading_density','overlap_fraction','overlap_peak','rhythm_entropy',
                                         'rhythm_switch_rate','offgrid_fraction','tempo_change_rate',
                                         'tap_rate_median','tap_rate_p90','tap_rate_p99','movement_rate_p90','direction_change'),0.)
    radius=max(1.,54.4-4.48*cs)
    density=[];overlaps=[];start=0
    for i,(t,x,y) in enumerate(objects):
        while start<i and t-objects[start][0]>preempt_ms:start+=1
        density.append(i-start+1)
        overlaps.append(sum((x-ox)**2+(y-oy)**2<(2*radius)**2 for _,ox,oy in objects[start:i]))
    offsets=[t[0] for t in timings]
    ratios=[];offgrid=0
    subdivisions=(1/16,1/12,1/8,1/6,1/4,1/3,1/2,2/3,3/4,1.,1.5,2.,3.,4.)
    for previous,current in zip(objects,objects[1:]):
        gap=current[0]-previous[0]
        if not 20<=gap<=2000:continue
        idx=max(0,bisect_right(offsets,previous[0])-1)
        beat=timings[idx][1] if timings else 500/clock_rate
        ratio=gap/beat
        nearest=min(range(len(subdivisions)),key=lambda k:abs(math.log(max(ratio,1e-9)/subdivisions[k])))
        offgrid+=abs(ratio/subdivisions[nearest]-1)>.08
        ratios.append(nearest)
    counts=Counter(ratios);n=max(1,len(ratios))
    entropy=-sum((v/n)*math.log2(v/n) for v in counts.values())
    first,last=objects[0][0],objects[-1][0]
    changes=sum(first<=b[0]<=last and abs(a[1]/b[1]-1)>.02 for a,b in zip(timings,timings[1:]))
    ordered=sorted(density)
    def quantile(values,q):
        ordered=sorted(values)
        return float(ordered[min(len(ordered)-1,int(q*(len(ordered)-1)))]) if ordered else 0.
    rates=[];movements=[];directions=[]
    for a,b in zip(objects,objects[1:]):
        gap=b[0]-a[0]
        if not 20<=gap<=2000:continue
        rates.append(1000/gap)
        movements.append(math.hypot(b[1]-a[1],b[2]-a[2])/radius*1000/gap)
    for a,b,c in zip(objects,objects[1:],objects[2:]):
        u=(b[1]-a[1],b[2]-a[2]);v=(c[1]-b[1],c[2]-b[2])
        norm=math.hypot(*u)*math.hypot(*v)
        if norm and max(b[0]-a[0],c[0]-b[0])<1000:
            directions.append(math.acos(max(-1.,min(1.,(u[0]*v[0]+u[1]*v[1])/norm)))/math.pi)
    return {'reading_density':float(ordered[min(len(ordered)-1,int(.9*len(ordered)))]),
            'overlap_fraction':sum(v>0 for v in overlaps)/len(overlaps),
            'overlap_peak':float(max(overlaps)), 'rhythm_entropy':entropy,
            'rhythm_switch_rate':sum(a!=b for a,b in zip(ratios,ratios[1:]))/max(1,len(ratios)-1),
            'offgrid_fraction':offgrid/n,
            'tempo_change_rate':changes/max((last-first)/60000,1/60),
            'tap_rate_median':quantile(rates,.5),'tap_rate_p90':quantile(rates,.9),'tap_rate_p99':quantile(rates,.99),
            'movement_rate_p90':quantile(movements,.9),'direction_change':sum(directions)/max(1,len(directions))}
