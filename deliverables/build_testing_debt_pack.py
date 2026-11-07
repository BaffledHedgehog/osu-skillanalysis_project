"""Prepare the separate testing homework pack (lecture 5, slides 20-26)."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile

from docx import Document
from docx.shared import Cm, Pt
from build_practice_tests import CASES, table

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/testing-debt-2026-11-07'
RUN=ROOT/'.data/testing-debt-2026-11-07'
SOURCE='https://drive.google.com/file/d/1kziIv7DTdeVkrQdYPza9egt-qMDVcbGv/view'
DATE='07.11.2026'


def document(title):
    d=Document();s=d.sections[0]
    s.top_margin=s.bottom_margin=Cm(1.5)
    s.left_margin=s.right_margin=Cm(1.8)
    d.styles['Normal'].font.name='Calibri';d.styles['Normal'].font.size=Pt(10)
    d.styles['Normal'].paragraph_format.space_after=Pt(5)
    d.add_heading(title,0)
    d.add_paragraph('pulse! | Долг по тестированию | Проверено '+DATE+' | Ответственный: Темиров А.М.')
    return d


def load(case,name):
    return json.loads((RUN/'evidence'/case/name).read_text(encoding='utf-8'))


def make_plan():
    d=document('Тест-план MVP')
    table(d,[
        ('Объект и версия','Локальный MVP pulse!: основа 3bfd0ca и рабочие изменения 07.11. Python 3.13.5, FastAPI, SQLite; pytest 9.0.3. Автотесты создают отдельные базы.'),
        ('Цель','Проверить импорт карты, выбор модели по результату проверки и поведение при недостатке истории.'),
        ('Входит','REQ-02: .osu/.osz и отклонение поврежденного файла. REQ-06: Brier модели < baseline. Дополнительно REQ-05/09: нет выдуманного прогноза при двух результатах.'),
        ('Не входит','Качество на всех игроках, полный набор 16 навыков, нагрузка публичного сервиса, платежи, нативный разбор реплеев и HP.'),
        ('Риски и проверки','Неверный формат, изменение каталога при ошибке, неверное включение ML, фиктивный прогноз. Функциональные, негативные, граничные проверки API и интерфейса.'),
        ('Начало и завершение','Вход: зависимости установлены, карта-образец доступна. Выход: 6 кейсов Pass, факты и evidence сохранены, нет блокирующей ошибки в выбранном сценарии; известные проблемы перечислены.'),
        ('Данные и метод','Карты из test-kit; изолированные синтетические истории. Brier 0.12/0.20/0.28 заданы как контрольные условия. Они не являются оценкой качества на игроках.'),
        ('Ответственность','QA/Темиров: прогон и отчет. DEV/Гайфуллин: импорт и UI. ML/Недугов: модели и метрики. PM/Ляхов: объем проверки и итоговое решение.'),
    ])
    d.add_paragraph('Основание: лекция 5, слайд 14 (план), 20 (дизайн кейсов), 26 (минимум 6 кейсов в ДЗ).')
    d.save(OUT/'01 - Тест-план.docx')


def make_cases():
    cases=list(CASES)+[('TC-06','REQ-05, REQ-09','Недостаточно истории для прогноза','Негативный',
        'Изолированный MVP; QA_TwoResults имеет ровно 2 результата. Загружены карта истории 600001 и целевая 600002. Минимум для первого прогноза - 3 результата.',
        '1. Создать 2 завершенных результата.\n2. Запросить анализ карты 600002 для QA_TwoResults.\n3. Проверить значения прогноза, сообщение и health.',
        'HTTP 200, карта анализируется. pass_probability и expected_accuracy равны null, accuracy_range=null, model=insufficient_data. Сообщение объясняет нехватку истории. Приложение отвечает.',
        'evidence/TC-06/analysis.json, conditions.json; practice.xml; pytest.txt.')]
    a=load('TC-01','valid-map.osu-analysis.json')
    actual={
        'TC-01':f"Оба формата приняты. Карта {a['id']}, {a['object_count']} объектов, {a['stars']} звезд. ID, название и признаки сверены. Повтор той же карты не создает второй ID.",
        'TC-02':'Контрольные Brier 0.12 < 0.20. Статус trained, источник validated_personal_logistic; целевая карта отсутствует в истории. Обучающие и проверочные карты не пересекаются.',
        'TC-03':'HTTP 400, понятный текст ошибки. Список ID каталога до и после совпал. /api/health вернул 200.',
        'TC-04':'Контрольные Brier 0.28 > 0.20. Статус rejected, используется similarity_and_prior, не проверенная ML прохождения.',
        'TC-05':'Контрольные Brier 0.20 == 0.20. Статус rejected. Равенство не считается улучшением.',
        'TC-06':'При двух результатах обе числовые оценки и интервал отсутствуют. model=insufficient_data, observations=2. Анализ карты и health доступны.',
    }
    d=document('6 тест-кейсов с результатами')
    d.add_paragraph('2 успешных, 3 негативных, 1 граничный. REQ-02 и REQ-06 - два выбранных Must. TC-06 дополнительно проверяет условия REQ-05/09. Прогон выполнен автоматически, ручное участие человека не заявляется.')
    for i,(cid,req,title,kind,pre,steps,expected,_) in enumerate(cases):
        if i:d.add_page_break()
        d.add_heading(cid+'. '+title,1)
        evidence='evidence/'+cid+'/; practice.xml; pytest.txt'
        if cid in ['TC-02','TC-04','TC-05']:evidence+='; conditions.json содержит контрольные метрики, training.json - решение, analysis.json - источник прогноза.'
        table(d,[('TC-ID / REQ-ID',cid+' / '+req),('Тип',kind),('Предусловия',pre),
                 ('Шаги',steps),('Ожидаемый результат',expected),('Фактический результат',actual[cid]),
                 ('Статус','Pass'),('Ответственный','Темиров А.М. / QA'),('Доказательства',evidence)])
    d.save(OUT/'02 - Шесть тест-кейсов.docx')


def make_bugs():
    d=document('Журнал дефектов и разбор результатов')
    d.add_paragraph('Статусы исходных проблем сохранены. Успешный текущий прогон не означает, что все старые дефекты исправлены. Следующие даты ниже - план повторной проверки, не результат проверки.')
    issues=[
        ('NATIVE-01','Продукт / дополнительная функция','Таймаут проверки отключения звука в .NET host 04.11. Обработка реплеев не стартовала.',
         'Нужно безопасно подтвердить отключение звука и выполнить проверку; фактически таймаут 30 с.',
         'Medium / P2','Гайфуллин А.Р. / DEV','Открыт; нативные HP/реплеи не включены в эти 6 кейсов.',
         '08.11.2026','Исследовать отдельно. Базовый анализ карты работает; небезопасный запуск не обходить.',
         'historical/native-audio.xml и native-audio.txt'),
        ('TEST-01','Дефект теста, не приложения','В файлах команды TC-05 отмечен Fail: assert 0.273 == 0.2509.',
         'Нужно задать равные метрики; фактически проверялись разные значения, до проверки rejected не дошли.',
         'Low / P2','Темиров А.М. / QA','Проверка заменена корректным сценарием; TC-05 повторно Pass. Исходники команды сохранены.',
         '07.11.2026, повтор выполнен','Не исправлять приложение по этому скриншоту. Контролируемое равенство 0.20/0.20 отклоняет модель.',
         'evidence/TC-05/; historical/TC-05_1.png, TC-05_2.png'),
        ('PERF-01','Наблюдение, требует воспроизведения','04.11 анализ во время импорта не ответил за 40 с; повтор после импорта ответил за 0.1 с.',
         'Причина и повторяемость не установлены; строгий критерий конкурентной нагрузки не согласован.',
         'Medium предварительно / P2','Гайфуллин А.Р. / DEV','Открытое наблюдение. Не выдаем повторный быстрый ответ за исправление.',
         '08.11.2026','Отдельно повторить одновременный импорт и анализ с замером. Зафиксировать среду и критерий.',
         'historical/test-results-2026-11-04.json, запись PERF-01'),
        ('QUALITY-01','Наблюдение качества модели','Для 4610937 старый прогноз accuracy 82.3%, игрок сообщил 94.01%; ошибка 11.71 п.п.',
         'Одна попытка не доказывает качество всей модели и не задает универсальный предел ошибки.',
         'Medium / P2','Недугов Д. / ML','Метод улучшен; исправление этого случая на будущих играх не подтверждено.',
         '08.11.2026','Для следующей попытки заранее сохранить прогноз. Не заменять старый прогноз пересчетом после игры.',
         'historical/test-results-2026-11-04.json, запись QUALITY-01'),
    ]
    for issue in issues:
        cid,kind,title,fact,severity,owner,status,date,decision,evidence=issue
        d.add_heading(cid+'. '+kind,1)
        table(d,[('Факт',title+'\n'+fact),('Severity / Priority',severity),('Владелец',owner),
                 ('Статус',status),('Следующая проверка',date),('Решение',decision),('Evidence',evidence)])
    d.add_paragraph('По результатам выбранных 6 кейсов блокирующие дефекты не обнаружены. Это вывод о заданном сценарии, а не о всем продукте. NATIVE-01 остается открытым за пределами текущего объема.')
    d.save(OUT/'03 - Журнал дефектов.docx')


def make_report():
    suites=ET.parse(RUN/'practice.xml').getroot().findall('testsuite')
    counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    assert counts=={'tests':6,'failures':0,'errors':0,'skipped':0},counts
    d=document('Отчет о тестировании')
    table(d,[('Версия','Основа кода 3bfd0ca и локальные изменения 07.11. Прогон 6 учебных кейсов, отдельные временные базы.'),
             ('Результат','Всего 6. Passed 6 / Failed 0 / Blocked 0 / Skipped 0. Один warning зависимости не является Fail.'),
             ('Выбранные Must','REQ-02 и REQ-06 проверены: оба формата, поврежденный файл, модель лучше/хуже/равна базовой. Это 2 выбранных Must, не заявление о 100% всех требований ТЗ.'),
             ('Дополнительно','TC-06: при двух результатах нет выдуманных pass/accuracy. Связь с REQ-05 и REQ-09.'),
             ('Доказательства','practice.xml, pytest.txt, evidence/TC-01..06. Контрольные Brier отдельно помечены как данные тестов.'),
             ('Дефекты','NATIVE-01 открыт вне текущего scope. TEST-01 разобран, корректный TC-05 Pass. PERF-01 и QUALITY-01 требуют отдельной проверки.'),
             ('README другим участником','Файлы new/ подтверждают ручной запуск копии у коллег. Полный новый Quick Start с точной версией другим человеком не зафиксирован. Перед сдачей повторить и сохранить результат.'),
             ('Рекомендация','Работу по этим 6 кейсам представить к проверке с журналом ограничений. Решение о зачете долга принимает преподаватель; автоматическая приемка не объявлена.')])
    d.add_heading('Что показать на занятии',1)
    for text in ['Тест-план: выбранный scope и критерии завершения.',
                 'Кейсы: TC-01, TC-03 и TC-05; затем лог 6 passed.',
                 'Для TC-05: training.json с двумя Brier 0.20 и rejected.',
                 'Bug Log: статус NATIVE-01 и разбор ошибки постановки TEST-01.',
                 'Работающий сайт и результат запуска другим участником по README.']:
        d.add_paragraph('- '+text)
    d.save(OUT/'04 - Отчет о тестировании.docx')


def make_updates():
    d=document('Связь требований, рисков и решений')
    d.add_heading('REQ / AC / TC',1)
    table(d,[('REQ-02 / Must','AC: верные .osu и .osz приняты, поля совпадают. Неверный файл отклонен без изменения каталога. TC-01, TC-03: Pass.'),
             ('REQ-06 / Must','AC: ML pass включается только если Brier модели строго ниже baseline на проверочных картах. TC-02, TC-04, TC-05: Pass.'),
             ('REQ-05 / REQ-09','Дополнительное условие: при 2 результатах, когда минимум равен 3, числовые прогнозы не выдумываются. TC-06: Pass.')])
    d.add_heading('Решения по проверке',1)
    table(d,[('D-T01 / 07.11','Расширить 5 практических кейсов до 6: слайд 26 требует не менее 6. Добавлен TC-06, повторный прогон 6 passed.'),
             ('D-T02 / 07.11','Отделить контроль Brier от качества модели. Эти тесты проверяют выбор алгоритма, а не качество вероятностей на игроках.'),
             ('D-T03 / 07.11','Не считать неравные числа равенством. Спорный Fail команды не подтверждает баг приложения; корректный граничный тест Pass.'),
             ('D-T04 / 07.11','Сохранить открытые проблемы в Bug Log. Нативный разбор не запускать ради демонстрации этих кейсов.')])
    d.add_heading('Обновление задач',1)
    table(d,[('Готово / QA','TC-06 добавлен, 6 кейсов выполнены, факты и evidence сохранены.'),
             ('Готово / QA','Тест-план, кейсы, Bug Log, Test Report и памятка собраны отдельно от итоговой презентации.'),
             ('Проверить / QA + участник','Повторить Quick Start другим человеком, записать имя, версию, команду и результат. Не ставить Pass заранее.'),
             ('Отдельно / DEV + ML','Разобрать NATIVE-01, воспроизвести PERF-01, проверить следующие прогнозы QUALITY-01.')])
    d.add_heading('Риски',1)
    table(d,[('R-T01 / QA','Если показать только 5 кейсов, состав ДЗ не выполнен. Триггер: меньше 6 в журнале. Мера выполнена: TC-06, 6 passed.'),
             ('R-T02 / ML','Если выдать заданные Brier за реальное качество, вывод о модели будет ложным. Триггер: смешаны MAE и Brier. Мера: явные подписи и отдельные отчеты.'),
             ('R-T03 / DEV','Если запуск зависит от окружения автора, сосед не воспроизведет результат. Триггер: ошибка Quick Start. Мера: отдельное venv проверено автоматически, повтор человеком записывается отдельно.')])
    d.save(OUT/'05 - Требования и решения.docx')


GUIDE='''# Как закрыть долг по тестированию

Это отдельное задание, не текст итоговой защиты проекта. Комплект подготовлен 07.11.2026.

## Что требует лекция

Слайд 20: на практике 5 кейсов. Слайд 26: в домашней работе уже не менее 6, обязательно успешные, негативные и граничные. Еще нужны Test Plan, Bug Log, Test Report, запуск другим участником по README, обновленные требования и решения.

## Что сдать

1. 01 - Тест-план.docx: одна страница с объемом проверки и условиями завершения.
2. 02 - Шесть тест-кейсов.docx: ID, требования, данные, шаги, ожидание, факт, ответственный и evidence.
3. 03 - Журнал дефектов.docx: реальные проблемы, статус, severity/priority, владелец и следующий шаг.
4. 04 - Отчет о тестировании.docx: 6 passed, 0 failed, 0 blocked; границы проверки.
5. 05 - Требования и решения.docx: связь REQ/AC/TC, задачи, риски и решения.
6. evidence/, practice.xml и pytest.txt: доказательства прогонов. Скриншоты или логи допустимы; не нужно делать скриншот каждого клика.
7. README и результат повторного запуска другим участником. Инструкция скопирована в Quick Start.md.

## Что осталось сделать человеку

Один другой участник запускает проект по Quick Start на своем компьютере. Загружает карту из inputs/, открывает анализ и сохраняет скриншот результата. Заполняет "Запуск другим участником.md": кто, когда, версия, команды, результат, путь к скриншоту. Скриншоты коллег из new/ уже полезны, но не фиксируют весь путь по новой инструкции. Автоматическую чистую установку мы проверили, выдавать ее за действия другого человека нельзя.

## Что показать за 3-5 минут

1. Открыть план и сказать, что проверялись REQ-02 и REQ-06.
2. Показать таблицу 6 кейсов, затем pytest.txt с 6 passed.
3. Открыть evidence/TC-05/training.json: brier=0.20, baseline_brier=0.20, status=rejected.
4. Показать TC-03: HTTP 400, списки каталога до/после одинаковы.
5. Открыть Bug Log: проблемы не скрыты, у каждой есть владелец и решение.
6. Показать результат запуска другим участником. Если не успели, прямо назвать этот незакрытый пункт.

## Текст, который можно сказать

"Мы закрываем отдельное задание по тестированию. Выбрали два Must-требования: импорт карты и включение модели только при улучшении относительно базовой оценки. Подготовили шесть кейсов: два успешных, три негативных и один граничный. Все шесть прошли, результаты и ответы API сохранены.

Отдельно проверили равенство ошибок: при 0.20 против 0.20 модель отклоняется. В старом отчете этот кейс был помечен Fail, но там сравнивали разные числа. Исправили постановку проверки. Контрольные Brier проверяют логику приложения, а не качество модели на игроках.

В журнале сохранены известные проблемы, включая дополнительный нативный разбор реплеев. Он не входит в объем этих шести кейсов. Для выбранного сценария блокирующих ошибок не обнаружено. Передаем план, кейсы, журнал, отчет и доказательства."

После реального запуска соседом можно добавить: "Запуск по README повторил [имя], результат приложен". До этого так говорить нельзя.

## Команда повторного прогона

Из корня репозитория:

```powershell
python -m pip install -r requirements.txt pytest
python -m pytest tests/test_practice_cases.py -v
```

Это шесть оформленных учебных кейсов. Число 149 в итоговой презентации относится к более широкому предыдущему прогону проекта, а не к числу кейсов ДЗ. Для записи нового evidence использовать run-six-cases.ps1 из этой папки; он создает новую папку прогона и сохраняет прошлые доказательства.

## Про текст команды

Для итоговой сдачи используй обновленный "Что говорить и показывать" из defense-2026-11-07: там исправлены даты и учтены личные шкалы. Старую речь команды можно взять за основу, но читать ее без правок не стоит. Для долга по тестам достаточно короткого текста выше.

Источник: лекция 5, слайды 18, 20, 23-26.
'''


def make_guide():
    (OUT/'Что сдать и говорить.md').write_text(GUIDE+'\nЛекция: '+SOURCE+'\n',encoding='utf-8')
    d=document('Что сдать и показать по тестированию')
    for line in GUIDE.splitlines():
        if line.startswith('## '):d.add_heading(line[3:],1)
        elif line.startswith('# '):continue
        elif line:d.add_paragraph(line.replace('`',''))
    d.add_paragraph('Лекция: '+SOURCE);d.save(OUT/'Что сдать и говорить.docx')
    (OUT/'Запуск другим участником.md').write_text('''# Запуск по README другим участником

Статус: не заполнено. Не ставить Pass до выполнения.

- Участник:
- Дата и время:
- Версия кода: git rev-parse --short HEAD и наличие локальных изменений.
- Python: python --version.
- Команды установки и запуска:
- Какие действия выполнены: запуск, загрузка .osu/.osz, открытие анализа.
- Фактический результат:
- Статус: Pass / Fail / Blocked.
- Скриншот или лог, путь:
- Замечания к инструкции:
''',encoding='utf-8')
    (OUT/'run-six-cases.ps1').write_text('''$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $projectRoot
$runStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$evidencePath = Join-Path $projectRoot ('.data/testing-debt-runs/' + $runStamp)
New-Item -ItemType Directory -Path $evidencePath -Force | Out-Null
$previousEvidence = $env:PULSE_TEST_EVIDENCE_DIR
try {
    $env:PULSE_TEST_EVIDENCE_DIR = Join-Path $evidencePath 'evidence'
    python -m pytest tests/test_practice_cases.py -v --junitxml="$evidencePath/practice.xml" 2>&1 | Tee-Object -FilePath "$evidencePath/pytest.txt"
    if ($LASTEXITCODE -ne 0) { throw 'A case failed. See the saved log.' }
} finally {
    $env:PULSE_TEST_EVIDENCE_DIR = $previousEvidence
}
Write-Output $evidencePath
''',encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    make_plan();make_cases();make_bugs();make_report();make_updates();make_guide()
    shutil.copytree(RUN/'evidence',OUT/'evidence',dirs_exist_ok=True)
    for file in ['practice.xml','pytest.txt']:shutil.copy2(RUN/file,OUT/file)
    shutil.copytree(ROOT/'deliverables/test-kit/inputs',OUT/'inputs',dirs_exist_ok=True)
    shutil.copy2(ROOT/'docs/quick_start.md',OUT/'Quick Start.md')
    for name in ['test_practice_cases.py','test_app.py','conftest.py']:
        target=OUT/'tests'/name;target.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT/'tests'/name,target)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    names=['tests/test_practice_cases.py','tests/test_app.py','tests/conftest.py',
           'app/main.py','app/ml.py','app/predictor.py','app/skill_profile.py']
    manifest={'date':DATE,'git_head':head,'working_tree_changes_included':True,
              'test_command':'python -m pytest tests/test_practice_cases.py -v',
              'passed':6,'failed':0,'blocked':0,
              'sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}}
    (OUT/'run-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    historical=OUT/'historical';historical.mkdir(exist_ok=True)
    shutil.copy2(ROOT/'deliverables/test-kit/test-results.json',historical/'test-results-2026-11-04.json')
    for name in ['native-audio.xml','native-audio.txt']:
        source=ROOT/'.data/test-evidence/2026-11-04'/name
        if source.exists():shutil.copy2(source,historical/name)
    with zipfile.ZipFile(ROOT/'new/test_results.zip') as z:
        for name in ['TC-05_1.png','TC-05_2.png']:(historical/name).write_bytes(z.read('test_results/'+name))
    (OUT/'README.md').write_text('''# Отдельный долг по тестированию

Начать с "Что сдать и говорить.docx". По лекции 5 в ДЗ нужно не менее 6 кейсов, а не только 5 из практики.
Автоматически: 6 passed / 0 failed / 0 blocked. Еще требуется запись реального повторного запуска по инструкции другим участником.
Это не презентация итоговой защиты. Итоговая досрочная сдача проекта - 07.11.2026; ее материалы лежат отдельно в ../defense-2026-11-07/.
''',encoding='utf-8')
    archive=ROOT/'deliverables/pulse-testing-debt-2026-11-07.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file in OUT.rglob('*'):
            if file.is_file():z.write(file,file.relative_to(OUT))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(OUT);print(archive)


if __name__=='__main__':main()
