"""Summarise extra regressions and read-only live/browser checks."""
from pathlib import Path
import json
import shutil
import xml.etree.ElementTree as ET

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import build_course_audit as layout

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'.data/extended-tests-2026-11-07'
OUT=ROOT/'deliverables/extended-testing-2026-11-07'


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    suite=ET.parse(SOURCE/'tests.xml').getroot()[0]
    counts={k:int(suite.get(k,0)) for k in ['tests','failures','errors','skipped']}
    passed=counts['tests']-counts['failures']-counts['errors']-counts['skipped']
    browser=json.loads((SOURCE/'browser-after/result.json').read_text())
    smoke=json.loads((SOURCE/'real-map-smoke.json').read_text())
    assert counts['failures']==counts['errors']==0 and browser['passed']
    layout.OUT=OUT
    for name,file in [('CourseArial','arial.ttf'),('CourseBold','arialbd.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
    rows=[
        ['EX-01','Импорт lazer, затем stable того же результата','Одна попытка; accuracy остается lazer, а не legacy','Passed'],
        ['EX-02','Поврежденные настройки поправки: NaN/Inf/0/отрицательное значение','Поправка отключается, используется базовый прогноз','Passed'],
        ['EX-03','Нулевая нагрузка aim на карте при известном профиле','Нагрузка 0 учитывается как 0; индекс сложности остается доступен','Passed'],
        ['EX-04','Быстро выбрать HD, затем NM; задержать ответ HD','После обоих ответов выбран и отображен NM','Passed'],
        ['EX-05','Быстро выбрать две разные карты; задержать первую','Показана последняя выбранная карта','Passed'],
        ['EX-06','Переключить разминку на вызов; задержать первый ответ','Показан подбор для цели вызов','Passed'],
    ]
    layout.render('Дополнительные проверки - pulse!','Дополнительные проверки pulse!',[
        ('Версия и область', 'Проверено 07.11.2026 после коммита 08400e9, с текущими изменениями исходников. Изменяющие данные проверки выполнены в изолированных временных базах. Работающие серверы 8011 и 8012 не перезапускались.'),
        ('Общий результат', f'{passed} Passed, {counts["failures"]} Failed, {counts["errors"]} Errors, {counts["skipped"]} Skipped. Нативный тест аудио не запускался. Три браузерные проверки очередности ответов прошли без ошибок JavaScript.'),
        ('Проверки и исправления', (['ID','Что проверено','Ожидаемый и фактический результат','Итог'],rows)),
        ('Реальные карты', (['Группа','Карта / моды','Результат'],[
            [r['category'],r['map_id']+' / '+r['mods'],r['status']+': accuracy в диапазоне, интервал содержит оценку; механические aim/speed совпадают для NM/HD'] for r in smoke['cases']
        ])),
        ('Что эти проверки подтверждают', 'Проверено корректное поведение на простых, сложных и низко-AR картах, а также при задержке ответов и неверном артефакте. Новых результатов игры для проверки качества не собирали. Это не новая оценка MAE или калибровки pass и не независимая проверка двух карт, использованных для личного уточнения.'),
        ('Состояние серверов', 'Оба /api/health вернули HTTP 200. Статические исправления интерфейса доступны после обновления страницы. Для применения серверных исправлений потребуется следующий запуск соответствующего процесса; во время подготовки защиты процессы оставлены работать.'),
        ('Доказательства и повторение', 'tests.xml, browser-after.json, real-map-smoke.json рядом с отчетом. Команды из корня: python -m pytest -q; python tools/verify_browser_races.py --output .data/browser-repeat. Для браузерного скрипта нужны Playwright и Chrome. Регрессионные проверки импорта и поврежденного артефакта находятся в tests/.'),
    ])
    shutil.copy2(SOURCE/'tests.xml',OUT/'tests.xml')
    shutil.copy2(SOURCE/'browser-after/result.json',OUT/'browser-after.json')
    shutil.copy2(SOURCE/'browser-before/result.json',OUT/'browser-before.json')
    shutil.copy2(SOURCE/'real-map-smoke.json',OUT/'real-map-smoke.json')
    print(OUT)


if __name__=='__main__':build()
