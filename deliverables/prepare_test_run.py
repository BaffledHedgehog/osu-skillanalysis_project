"""Prepare inputs, an isolated database and a manual test guide. No game/audio run."""
import hashlib
import json
import csv
from pathlib import Path
import sqlite3
import zipfile
import xml.etree.ElementTree as ET

from docx import Document
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[1]
DAY = '2026-11-04'
KIT = ROOT / 'deliverables/test-kit'
EVIDENCE = ROOT / '.data/test-evidence' / DAY
STAND = ROOT / '.data' / f'test-stand-{DAY}'


def prepare():
    inputs = KIT / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / 'manual').mkdir(parents=True, exist_ok=True)
    STAND.mkdir(parents=True, exist_ok=True)
    source = ROOT / '.data/demo-map-4864861.osu'
    if not source.is_file():
        source = inputs / 'valid-map.osu'
    if not source.is_file():
        raise FileNotFoundError('Need demo-map-4864861.osu or test-kit/inputs/valid-map.osu')
    raw = source.read_bytes()
    (inputs / 'valid-map.osu').write_bytes(raw)
    with zipfile.ZipFile(inputs / 'valid-map.osz', 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('valid-map.osu', raw)
    (inputs / 'broken-map.osu').write_text('This is not an osu beatmap.\n', encoding='ascii')

    from app.analyzer import parse_map, analyze_map
    parsed = parse_map(raw)
    analysis = analyze_map(parsed)
    expected = {key: analysis[key] for key in ('id','artist','title','version','object_count','stars','aim','speed')}
    expected['mode'] = 'osu!standard (Mode: 0)'
    expected['sha256'] = hashlib.sha256(raw).hexdigest()
    (KIT / 'expected-map.json').write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding='utf-8')

    database = STAND / 'test.db'
    if not database.exists():
        original = ROOT / '.data/osu_skill.db'
        if original.is_file():
            with sqlite3.connect(original.resolve().as_uri() + '?mode=ro', uri=True) as src, sqlite3.connect(database) as dst:
                src.backup(dst)
        else:
            from app.storage import Store
            Store(database)

    launch = '''$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $projectRoot
$previousDatabase = $env:DATABASE_PATH
$previousPublicMode = $env:PULSE_PUBLIC_MODE
try {
    $env:DATABASE_PATH = Join-Path $projectRoot '.data/test-stand-2026-11-04/test.db'
    $env:PULSE_PUBLIC_MODE = '0'
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
} finally {
    $env:DATABASE_PATH = $previousDatabase
    $env:PULSE_PUBLIC_MODE = $previousPublicMode
}
'''
    (KIT / 'start-test-stand.ps1').write_text(launch, encoding='utf-8')
    run = '''$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $projectRoot
$evidencePath = Join-Path $projectRoot '.data/test-evidence/2026-11-04'
New-Item -ItemType Directory -Path $evidencePath -Force | Out-Null
$previousEvidence = $env:PULSE_TEST_EVIDENCE_DIR
try {
    $env:PULSE_TEST_EVIDENCE_DIR = Join-Path $evidencePath 'automated'
    python -m pytest tests/test_practice_cases.py -v --junitxml="$evidencePath/practice.xml" 2>&1 | Tee-Object -FilePath "$evidencePath/pytest.txt"
    if ($LASTEXITCODE -ne 0) { throw 'Some cases failed. See pytest.txt.' }
} finally {
    $env:PULSE_TEST_EVIDENCE_DIR = $previousEvidence
}
'''
    (KIT / 'run-five-cases.ps1').write_text(run, encoding='utf-8')

    cases = [('TC-01','REQ-02','Успешный','Оба формата приняты, поля карты совпали.'),
             ('TC-02','REQ-06','Успешный','0.12 < 0.20: trained, ML прохождения.'),
             ('TC-03','REQ-02','Негативный','Ошибка 400, каталог не изменился, сервер работает.'),
             ('TC-04','REQ-06','Негативный','0.28 > 0.20: rejected, предварительный расчет.'),
             ('TC-05','REQ-06','Граничный','0.20 = 0.20: rejected, предварительный расчет.')]
    statuses = {}
    xml = EVIDENCE / 'practice.xml'
    if xml.is_file():
        for case in ET.parse(xml).findall('.//testcase'):
            name = case.get('name','')
            code = 'TC-01' if name.startswith('test_tc01') else 'TC-03' if name.startswith('test_tc03') else next((c[0] for c in cases if c[0] in name),None)
            if code:
                statuses[code] = 'Fail' if case.find('failure') is not None or case.find('error') is not None else 'Blocked' if case.find('skipped') is not None else 'Pass'
    journal = KIT / 'Журнал прогона.csv'
    # Do not replace a tester's manually entered observations on another run.
    if not journal.exists():
        with journal.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f,delimiter=';')
            writer.writerow(['TC-ID','REQ-ID','Тип','Ожидание','Статус автотеста','Факт ручного прогона','Статус ручного прогона','Evidence'])
            for code,req,kind,expected_result in cases:
                writer.writerow([code,req,kind,expected_result,statuses.get(code,'Не выполнен'),'',
                    'Не выполнен' if code in ('TC-01','TC-03') else 'Проверяется автотестом',str(EVIDENCE/'automated'/code)])

    doc = Document()
    section = doc.sections[0]
    section.left_margin = section.right_margin = Cm(2)
    style = doc.styles['Normal']
    style.font.name = 'Calibri'; style.font.size = Pt(11)
    doc.add_heading('Как выполнить тесты pulse!', 0)
    doc.add_paragraph('Университет Иннополис | Проект 2 | 04.11.2026')
    doc.add_paragraph('Основа: Тест-план MVP - pulse!.docx и Практика 1 - Тест-кейсы MVP - pulse!.docx. Первый документ содержит общий план и старые результаты. Во втором находятся TC-01..TC-05. Исторические результаты от 24 октября не заменяют новый прогон.')
    doc.add_heading('1. Перед началом', 1)
    doc.add_paragraph('Стенд: http://127.0.0.1:8010/?user=baffledhedgehog. Он использует копию базы .data/test-stand-2026-11-04/test.db. Основная база .data/osu_skill.db в ручном прогоне не меняется. Не запускай импорт локального клиента, cookie, HP или обучение CNN: для этих пяти кейсов они не нужны.')
    doc.add_paragraph('Если стенд остановлен, выполни из корня проекта: .\\deliverables\\test-kit\\start-test-stand.ps1. Держи этот терминал открытым. Проверка запуска: http://127.0.0.1:8010/api/health должен ответить 200 и показать ok: true.')
    doc.add_heading('2. TC-01: загрузить правильные файлы', 1)
    doc.add_paragraph('Нажми загрузку файла карты, выбери deliverables/test-kit/inputs/valid-map.osu. Открой анализ. Проверь ID 4864861, Turbo - PADORU / PADORU, сложность Ren\'s Extra и 126 объектов. Для NM текущая версия считает 5.57 stars, aim 3.04, speed 2.26. Это признаки карты, а не персональный прогноз.')
    doc.add_paragraph('Сохрани скриншот анализа. Загрузить valid-map.osz нужно отдельно тем же способом. Проверь те же поля и сохрани второй скриншот. В архиве та же карта; повтор ID в каталоге не нужен. Pass: оба формата приняты, данные совпали, ошибки нет.')
    doc.add_heading('3. TC-03: загрузить поврежденный файл', 1)
    doc.add_paragraph('До загрузки сохрани список ID: открой http://127.0.0.1:8010/api/maps и сохрани JSON. Загрузи deliverables/test-kit/inputs/broken-map.osu. Ожидается понятная ошибка неверного файла карты. Сохрани скриншот ошибки сразу, пока уведомление видно. Затем снова сохрани список ID каталога и проверь /api/health.')
    doc.add_paragraph('Pass: ошибка показана, список ID не изменился, сервер продолжает отвечать. Само сообщение об ошибке в этом кейсе является ожидаемым результатом.')
    doc.add_heading('4. TC-02, TC-04 и TC-05: проверить решение по Brier', 1)
    doc.add_paragraph('Эти кейсы не выполняются подбором карт на реальном аккаунте. В них заранее заданы контрольные ошибки для проверки ветвления программы. Выполни: .\\deliverables\\test-kit\\run-five-cases.ps1. Скрипт проверяет все пять кейсов, включая загрузку файлов через API. Ожидается 5 passed.')
    table = doc.add_table(rows=1, cols=4); table.style = 'Table Grid'
    for c,t in zip(table.rows[0].cells, ('Кейс','Brier модели','Базовый Brier','Ожидание')): c.text=t
    for values in [('TC-02','0.12','0.20','trained; ML прохождения'),('TC-04','0.28','0.20','rejected; предварительный расчет'),('TC-05','0.20','0.20','rejected; равенство не улучшение')]:
        for c,t in zip(table.add_row().cells,values): c.text=t
    doc.add_paragraph('В тесте реально выполняются обучение на тестовой базе, разделение карт и запрос прогноза через API. Заменяется только функция расчета Brier, чтобы получить заданные числа. Проверяется отсутствие общих карт обучения и проверки. Контрольные числа не доказывают качество модели на игроках.')
    doc.add_heading('5. Что записать и сохранить', 1)
    doc.add_paragraph('В исходном файле с пятью кейсами заполни "Фактический результат" и "Статус". Для TC-01 и TC-03 добавь собственные скриншоты после ручного прогона. Для TC-02, TC-04 и TC-05 используй соответствующие JSON и результат pytest. Не записывай ручную проверку как выполненную по чужому скриншоту.')
    doc.add_paragraph('Доказательства: .data/test-evidence/2026-11-04. В automated/TC-XX лежат ответы API и условия теста; practice.xml и pytest.txt содержат результаты прогона. Свои скриншоты положи в manual. Можно сохранить скриншот терминала с 5 passed. При Fail приложи шаги и факт ошибки; при Blocked запиши, чего не хватило для проверки.')
    doc.add_paragraph('Журнал прогона.csv рядом с этой инструкцией открывается в Excel. В нем результаты автотестов записаны отдельно от ручной проверки. Заполни пустые поля ручного факта после своих действий. Скриншоты browser-reference получены автоматически и служат примером.')
    doc.add_heading('6. Дополнительно по общему тест-плану', 1)
    doc.add_paragraph('Открой карту 4864861 на этом же стенде под своим профилем, сравни NM и DT: звезды и другие признаки должны измениться. Нажми "Обучить и проверить ML" и сохрани текущие статусы моделей. Качество accuracy оценивай по текущему отчету MAE, не по цифрам от 24 октября. Низкий объем данных или отказ модели не являются автоматически ошибкой приложения.')
    doc.add_paragraph('Для проверки живого импорта нажми "Обновить результаты профиля" и запиши числа recent/best, затем повтори импорт и проверь отсутствие дублей. Это отдельная проверка с интернетом. Ее нельзя считать выполненной только потому, что в копии базы уже есть история.')
    doc.add_paragraph('Профиль всех навыков и качество на будущих играх не подтверждаются этими пятью кейсами. Если новых попыток после сохранения модели нет, в отчете укажи отсутствие данных, а не Pass по качеству прогноза.')
    doc.add_heading('Что сказать на показе', 1)
    doc.add_paragraph('"Проверяем два обязательных требования: импорт карт и выбор модели по ошибке Brier. Два кейса показываем в интерфейсе. Три кейса с заданными Brier проверяем автоматически на отдельной базе. При ошибке модели ниже базовой используется ML; при худшей или равной ошибке используется предварительная оценка. Это проверка логики приложения. Качество прогнозов на будущих играх проверяется отдельно."')
    doc.save(KIT / 'Инструкция по тестам - pulse!.docx')
    print(json.dumps({'guide':str(KIT/'Инструкция по тестам - pulse!.docx'),'database':str(database),'expected_map':expected},ensure_ascii=True))


if __name__ == '__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    prepare()
