"""Aggregate cohort and model report. Never copy raw player databases to deliverables."""
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET

from docx import Document
from docx.shared import Cm,Pt
import xlsxwriter

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'.data/population'
OUT=ROOT/'deliverables'


def read(name):return json.loads((DATA/name).read_text(encoding='utf-8'))


def build():
    players=read('players.json');quality=read('quality.json');report=read('report.json');scope=read('status.json')
    groups=defaultdict(list)
    for p in players.values():groups[p['pp_bin']].append(p)
    sources={s:sum(p['sources'].get(s,{}).get('received',0) for p in players.values()) for s in ('recent','best','firsts')}
    with sqlite3.connect(DATA/'population.db') as db:
        raw_count=db.execute('SELECT count(*) FROM score_payloads').fetchone()[0]
        charts=db.execute('SELECT count(*) FROM maps').fetchone()[0]
        analyzable=db.execute('SELECT count(DISTINCT map_id) FROM plays').fetchone()[0]
    with_recent=sum(p['sources']['recent'].get('received',0)>0 for p in players.values())
    stats={'players':len(players),'with_recent':with_recent,'sources':sources,'raw_unique':raw_count,
           'ordinary_passes':quality['ordinary_passes'],'ordinary_fails':quality['ordinary_fails'],
           'normalized':quality['normalized'],'loaded_charts':charts,'analyzable_charts':analyzable,
           'chart_scope':scope,'quality':quality,'model':report}
    test_file=DATA/'test-results.xml'
    if test_file.exists():
        tests=ET.parse(test_file)
        stats['code_checks']={'passed':len(tests.findall('.//testcase'))-len(tests.findall('.//skipped'))-len(tests.findall('.//failure'))-len(tests.findall('.//error')),
                              'skipped':len(tests.findall('.//skipped')),'failed':len(tests.findall('.//failure'))+len(tests.findall('.//error'))}
    (OUT/'population-summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    doc=Document();section=doc.sections[0]
    section.left_margin=section.right_margin=Cm(1.8)
    section.top_margin=section.bottom_margin=Cm(1.6)
    doc.styles['Normal'].font.name='Calibri';doc.styles['Normal'].font.size=Pt(11)
    doc.add_heading('Датасет и общая модель - pulse!',0)
    doc.add_paragraph('Университет Иннополис | Учебный проект | 04.11.2026')
    doc.add_heading('Что собрано',1)
    doc.add_paragraph(f'Публичные результаты {len(players)} игроков из разных стран. Согласие всех участников подтверждено владельцем проекта в чате 04.11.2026. Индивидуальные подтверждения в репозиторий не загружались. Объем согласия в этом сборе: игровые результаты для обучения и проверки модели.')
    doc.add_paragraph(f"Получено {sum(sources.values())} записей в списках: recent {sources['recent']}, best {sources['best']}, firsts {sources['firsts']}. После объединения совпадающих score_id осталось {raw_count} уникальных результатов. Recent есть у {with_recent} игроков; у остальных доступны только отобранные успешные результаты.")
    doc.add_paragraph(f"После проверки форматов, модов и версий карт подготовлено {quality['normalized']} результатов. Обычных проходов: {quality['ordinary_passes']}, провалов: {quality['ordinary_fails']}. NF не превращается в проход без NF. Карты с несовпадающим checksum не используются как та же версия карты.")
    doc.add_paragraph('Источник: публичный API Hinamizawa. Pinned в этом источнике не поддержан. Recent ограничен доступной историей сайта. Куки, пароли и токены участников не нужны. Локальные базы и реплеи участников в этот сбор не входили.')
    doc.add_paragraph('Для пилота загружены все пригодные recent-карты и до 30 детерминированно выбранных карт best/firsts для контекста активного игрока. Уже кешированные общие карты переиспользованы. Все исходные результаты сохранены; ограничение касается загрузки карт для первого эксперимента.')
    doc.add_heading('Как обучалась модель',1)
    doc.add_paragraph('Прогноз pass: логистическая регрессия. Прогноз accuracy при успешном прохождении: Ridge. Вход: признаки карты, ее модов, ритма и пиков нагрузки, описание прежних результатов игрока и разница между нагрузкой карты и показанными навыками. Текущие pp, ник и место игрока не входят в признаки.')
    doc.add_paragraph('Целевые примеры берутся только из обычных recent. Best и firsts используются только как более ранние примеры показанных навыков. Результат целевой карты, равные по времени и будущие попытки исключены из контекста. Для профиля используются до 100 прежних обычных результатов и до 30 best/firsts; старые игры имеют меньший вес.')
    doc.add_paragraph('Игроки и семейства песен поделены на train, validation и test до обучения. Целевые игроки и песни не пересекаются между частями. Контекст строится только по более ранним играм на семействах train. Это проверка на сохраненной истории, а не доказательство точности на будущих играх.')
    doc.add_paragraph('При обучении диапазоны pp имеют равный суммарный вес, игроки внутри диапазона тоже имеют равный вес. При оценке каждый игрок имеет равный вес. Ошибки отдельно посчитаны по диапазонам pp. PP служат для набора и балансировки группы, а не заменяют метки pass и accuracy.')
    doc.add_heading('Результаты',1)
    if stats.get('code_checks'):
        checks=stats['code_checks']
        doc.add_paragraph(f"Проверки кода: {checks['passed']} passed, {checks['skipped']} skipped, {checks['failed']} failed. Это проверка работы сбора, разделения данных и расчета; качество модели описано ниже. Опциональный native host этим сбором не запускался.")
    for task,label in [('pass','Прохождение'),('accuracy','Accuracy при прохождении')]:
        task_report=report['tasks'][task]
        doc.add_heading(label,2)
        active=task_report.get('accepted',False)
        if active:
            doc.add_paragraph('Подключена для проверенных модов: '+', '.join(task_report['supported_mods']))
        elif task_report['status']=='insufficient_data':
            doc.add_paragraph('Не обучалась: в train не хватило обычных попыток или игроков. Best не использовались вместо недостающих меток.')
        else:
            reasons={'validation_too_few_players':'мало игроков в validation','test_too_few_players':'мало игроков в test',
                     'validation_too_few_samples':'мало попыток в validation','test_too_few_samples':'мало попыток в test',
                     'validation_too_few_outcomes':'мало проходов или провалов в validation','test_too_few_outcomes':'мало проходов или провалов в test',
                     'validation_not_better_than_baseline':'ошибка в validation не ниже базы','test_not_better_than_baseline':'ошибка в test не ниже базы',
                     'no_validated_mods':'нет модов с достаточной отдельной проверкой'}
            doc.add_paragraph('Обучена, веса сохранены. К прогнозу не подключена: '+ '; '.join(reasons.get(k,k) for k in task_report.get('release_reasons',[]))+'.')
        table=doc.add_table(rows=1,cols=6);table.style='Table Grid'
        for c,v in zip(table.rows[0].cells,['Часть','Игроки','Попытки','Pass / fail','Ошибка модели','Ошибка базы']):c.text=v
        for split in ('train','validation','test'):
            part=task_report.get('metrics',task_report.get('splits',{})).get(split,{})
            factor=1 if task=='pass' else 100
            error=part.get('model_error');baseline=part.get('baseline_error')
            cells=table.add_row().cells
            values=[split,str(part.get('players',0)),str(part.get('samples',0)),f"{part.get('passes',0)} / {part.get('fails',0)}",
                    '-' if error is None else f'{error*factor:.4f}', '-' if baseline is None else f'{baseline*factor:.4f}']
            for c,v in zip(cells,values):c.text=v
        doc.add_paragraph('Метрика: Brier, меньше лучше.' if task=='pass' else 'Метрика: MAE в процентных пунктах, меньше лучше.')
    doc.add_paragraph('Базовые прогнозы: постоянное среднее по train и оценка по прежней истории игрока. Для сравнения используется меньшая ошибка этих двух баз. Модель подключается только при улучшении в validation и test, достаточном числе игроков и примеров, отдельно для проверенных модов. Личная модель сохраняет приоритет на данных обычных попыток.')
    doc.add_heading('Что ограничивает результат',1)
    doc.add_paragraph('Список участников равномернее по pp, чем доступные обычные попытки. Пустой recent нельзя заменить выдуманными проходами из best. Одного снимка recent мало для ряда групп и модов. Следующий шаг - повторить сбор после новых игровых сессий и проверить новые прогнозы. Исходные результаты и веса хранятся локально в .data/population и не загружаются в публичный репозиторий.')
    doc.add_paragraph('Пересборка продолжает сохраненный сбор; для обновления сетевых списков нужен --force. Команды и схема данных описаны в POPULATION.md. Отчет JSON: deliverables/population-summary.json. Версия набора: '+report['fingerprint'][:16]+'.')
    doc.save(OUT/'Датасет и общая модель - pulse!.docx')
    book=xlsxwriter.Workbook(OUT/'Датасет и общая модель - pulse!.xlsx',{'strings_to_formulas':False})
    sheet=book.add_worksheet('Группы pp')
    columns=['Диапазон pp','Игроки','С recent','Recent','Провалы recent','Best','Firsts']
    sheet.write_row(0,0,columns)
    for i,band in enumerate(sorted(groups,key=lambda b:int(b.split('-')[0])),1):
        group=groups[band]
        sheet.write_row(i,0,[band,len(group),sum(p['sources']['recent'].get('received',0)>0 for p in group),
                            sum(p['sources']['recent'].get('received',0) for p in group),
                            sum(p['sources']['recent'].get('fails',0) for p in group),
                            sum(p['sources']['best'].get('received',0) for p in group),sum(p['sources']['firsts'].get('received',0) for p in group)])
    sheet.autofilter(0,0,len(groups),len(columns)-1);sheet.freeze_panes(1,1);sheet.set_column('A:G',20)
    sheet=book.add_worksheet('Проверка моделей');sheet.set_column('A:H',22)
    sheet.write_row(0,0,['Модель','Часть','Игроки','Попытки','Ошибка','Ошибка базы','Метрика','Подключена'])
    i=1
    for task,r in report['tasks'].items():
        for split,part in r.get('metrics',r.get('splits',{})).items():
            f=100 if task=='accuracy' else 1
            sheet.write_row(i,0,[task,split,part.get('players',0),part.get('samples',0),
                                part['model_error']*f if 'model_error' in part else '',part['baseline_error']*f if 'baseline_error' in part else '',
                                'MAE, п.п.' if task=='accuracy' else 'Brier','Да' if r.get('accepted') else 'Нет']);i+=1
    book.close()
    print(f'Population report written: {raw_count} unique scores, {quality["normalized"]} normalized')


if __name__=='__main__':build()
