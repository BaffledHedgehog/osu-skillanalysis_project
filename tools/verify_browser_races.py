"""Read-only browser tests with delayed fixture responses, no score imports.

Requires optional Playwright and Chrome. The live server is never restarted.
"""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from urllib.parse import urlparse,parse_qs

import httpx
from playwright.async_api import async_playwright


async def run(base,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fixtures={}
    async with httpx.AsyncClient(timeout=30) as c:
        for mid,mods in [('4610937','NM'),('4610937','HD'),('4934415','NM')]:
            r=await c.get(base+f'/api/maps/{mid}/analysis',params={'user':'BaffledHedgehog','mods':mods})
            r.raise_for_status();fixtures[mid,mods]=r.json()
    async with async_playwright() as p:
        browser=await p.chromium.launch(channel='chrome',headless=True,args=['--mute-audio'])
        page=await browser.new_page(viewport={'width':1280,'height':900})
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        phase='initial'
        async def route_api(route):
            nonlocal phase
            url=urlparse(route.request.url);q=parse_qs(url.query);path=url.path
            if path=='/api/maps':return await route.fulfill(json=[])
            if path=='/api/recommendations':
                goal=q.get('goal',['comfort'])[0]
                if phase=='goal':await asyncio.sleep(.35 if goal=='warmup' else .02)
                maps=[]
                if phase=='goal':
                    a=deepcopy(fixtures['4934415','NM']);a['title']=goal+' sentinel';maps=[a]
                return await route.fulfill(json={'maps':maps,'min_stars':0})
            if path.startswith('/api/maps/') and path.endswith('/analysis'):
                mid=path.split('/')[3];mods=q.get('mods',['NM'])[0]
                if phase=='mods':await asyncio.sleep(.35 if mods=='HD' else .02)
                if phase=='map':await asyncio.sleep(.35 if mid=='4610937' else .02)
                return await route.fulfill(json=fixtures[mid,mods])
            return await route.continue_()
        await page.route('**/api/**',route_api)
        await page.goto(base+'/?user=BaffledHedgehog&map=4610937',wait_until='networkidle')
        await page.wait_for_selector('#result-section:not(.hidden)')
        phase='mods'
        mods=await page.evaluate('''async () => {
          state.mods=new Set(['HD']);updateMods();const old=selectMap('4610937',false);
          await new Promise(resolve=>setTimeout(resolve,25));
          state.mods.clear();updateMods();const latest=selectMap('4610937',false);
          await Promise.all([old,latest]);
          return {selected:currentMods(),rendered:state.analysis.mods};
        }''')
        phase='map'
        maps=await page.evaluate('''async () => {
          const old=selectMap('4610937',false);await new Promise(resolve=>setTimeout(resolve,25));
          const latest=selectMap('4934415',false);await Promise.all([old,latest]);
          return {selected:state.mapId,rendered:state.analysis.id};
        }''')
        phase='goal'
        goal=await page.evaluate('''async () => {
          state.goal='warmup';const old=loadRecommendations();await new Promise(resolve=>setTimeout(resolve,25));
          state.goal='challenge';const latest=loadRecommendations();await Promise.all([old,latest]);
          return {selected:state.goal,rendered:document.querySelector('.rec-title').textContent};
        }''')
        report={'mods':mods,'maps':maps,'recommendations':goal,'browser_errors':errors}
        report['passed']=(mods['selected']==mods['rendered'] and maps['selected']==maps['rendered']
                          and goal['rendered']==goal['selected']+' sentinel' and not errors)
        await page.screenshot(path=str(output/'browser.png'))
        (output/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        await browser.close()
    print(json.dumps(report))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',default='http://127.0.0.1:8012')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    asyncio.run(run(args.base,args.output))
