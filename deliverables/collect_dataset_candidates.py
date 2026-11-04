"""Public recruitment candidates only. Does not contact players or collect scores."""
import asyncio
import csv
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import random
import httpx

ROOT=Path(__file__).resolve().parents[1]
COUNTRIES=['US','GB','RU','DE','FR','PL','KR','JP','CN','CA','AU','NZ','NL','FI','SE','LI','IS','MT','LU','IE','SG','HK','MY','PT','TR','BR','AR','CL','PE','ZA','IN','PH','ID','MX','UY']
BASE='https://mirror.hinamizawa.ai/v3/osu/rankings/osu/performance'


async def main():
    cache=ROOT/'.data/dataset-candidates';cache.mkdir(parents=True,exist_ok=True)
    semaphore=asyncio.Semaphore(2)
    async with httpx.AsyncClient(timeout=25,headers={'User-Agent':'pulse-osu-analysis/0.2 (Innopolis student project; consent recruitment)'}) as client:
        async def page(country):
            async with semaphore:
                params={'country':country} if country else {}
                try:
                    response=await client.get(BASE,params=params);response.raise_for_status();payload=response.json()
                    (cache/f'{country or "global"}.json').write_text(json.dumps(payload),encoding='utf-8')
                    await asyncio.sleep(.2)
                    return payload.get('data',{}).get('ranking',[]),str(response.url)
                except (httpx.HTTPError,ValueError) as exc:return [],str(exc)
        pages=await asyncio.gather(*(page(country) for country in ['',*COUNTRIES]))
    pool={}
    for rows,url in pages:
        for r in rows:
            if r.get('id') and r.get('username') and r.get('pp') is not None and r.get('play_count',0)>=200:
                pool.setdefault(r['id'],(r,url))
    rng=random.Random(20261104);selected=[];counts={}
    for lower in range(0,34000,2000):
        candidates=[v for v in pool.values() if lower<=float(v[0]['pp'])<lower+2000]
        rng.shuffle(candidates);country_count={};ordered=[]
        for row,url in candidates:
            country=row.get('country_code','')
            if country_count.get(country,0)<3:
                ordered.append((row,url));country_count[country]=country_count.get(country,0)+1
        used={r['id'] for r,_ in ordered};ordered.extend((r,u) for r,u in candidates if r['id'] not in used)
        counts[f'{lower}-{lower+2000}']={'available_in_pool':len(candidates),'primary':min(6,len(ordered))}
        for index,(row,url) in enumerate(ordered[:12]):
            selected.append({'pp_bin':f'{lower}-{lower+2000}','list':'primary' if index<6 else 'reserve',
                'username':row['username'],'pp':row['pp'],'user_id':row['id'],'country':row.get('country_code',''),
                'play_count':row.get('play_count'),'global_rank':row.get('global_rank'),
                'profile_url':f'https://osu.ppy.sh/users/{row["id"]}','source_url':url,
                'verified_at':datetime.now(timezone(timedelta(hours=3))).isoformat(),'consent':''})
    output=ROOT/'deliverables/dataset-candidates.csv'
    with output.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(selected[0]),delimiter=';');writer.writeheader();writer.writerows(selected)
    summary={'created_at':datetime.now(timezone(timedelta(hours=3))).isoformat(),'ruleset':'osu!standard',
        'pool_users':len(pool),'primary':sum(r['list']=='primary' for r in selected),'reserve':sum(r['list']=='reserve' for r in selected),
        'counts':counts,'method':'Fixed 2000 pp bins, target 6 primary and 6 reserve per bin. Seeded selection from global and 35 country first pages; recruitment pool is not a uniform random sample of all accounts. No contact or score collection performed.'}
    (ROOT/'deliverables/dataset-candidates-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=True))


if __name__=='__main__':asyncio.run(main())
