"""Three actual Must acceptance checks from the current isolated test run."""
from pathlib import Path
import json
import shutil
import zipfile
import xml.etree.ElementTree as ET

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import build_course_audit as layout

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'deliverables/acceptance-2026-11-07'


def build():
    layout.OUT = OUT
    for name, filename in [('CourseArial','arial.ttf'), ('CourseBold','arialbd.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts') / filename)))
    suites = ET.parse(OUT/'tests.xml').getroot().findall('testsuite')
    assert sum(int(s.get('tests',0)) for s in suites)==6
    assert all(int(s.get('failures',0))+int(s.get('errors',0))==0 for s in suites)
    def evidence(tc, filename):
        return json.loads((OUT/'evidence'/tc/filename).read_text(encoding='utf-8'))
    a=evidence('TC-01','valid-map.osu-analysis.json')
    prediction=evidence('TC-02','prediction.json')
    insufficient=evidence('TC-06','analysis.json')['prediction']
    assert insufficient['pass_probability'] is None
    sections=[
        ('Что проверяем', 'Выбраны ровно три Must-требования ТЗ: REQ-02, REQ-06, REQ-05. Код проверен повторно 07.11.2026 в отдельных тестовых базах. Все шесть связанных тестов прошли. Личная база игрока не менялась.'),
        ('Роли', (['Проверка', 'Кто отвечает за повтор', 'Кто принимает'],[
            ['AT-01 / REQ-02', 'Гайфуллин А.Р., разработка', 'Ляхов Я.П., заказчик'],
            ['AT-02 / REQ-06', 'Недугов Д., данные и модели', 'Ляхов Я.П., заказчик'],
            ['AT-03 / REQ-05', 'Темиров А.М., тестирование', 'Ляхов Я.П., заказчик'],
        ])),
        ('AT-01. Загрузка и анализ карты', [
            'REQ-02 / Must: приложение принимает .osu и .osz карты osu!standard и извлекает данные карты.',
            f"Вход: {a['id']} - {a['title']} [{a['version']}], {a['object_count']} объектов. Два формата одной карты: valid-map.osu и valid-map.osz. Отдельно - поврежденный файл broken-map.osu.",
            'AC-02: оба корректных формата возвращают HTTP 200, совпадают ID, название и число объектов, stars > 0. После повторной загрузки карты в каталоге один ее ID. Неверный файл возвращает HTTP 400 и не меняет каталог.',
            'Шаги: 1. Загрузить .osu. 2. Прочитать анализ и сравнить с исходником. 3. Загрузить .osz с той же картой и повторить сравнение. 4. Проверить каталог. 5. Загрузить поврежденный файл, сравнить каталог до и после.',
            f"Факт: оба корректных файла приняты, ID {a['id']}, объектов {a['object_count']}, stars {a['stars']}. Повтора ID нет. Поврежденный файл отклонен HTTP 400; каталог сохранен, health HTTP 200.",
            'Итог автоматической проверки: Passed. Замечания: нет в пределах проверяемого сценария. Покрытие: TC-01 и TC-03.',
            'Evidence: evidence/TC-01/valid-map.osu-upload.json, valid-map.osz-upload.json и оба файла analysis.json; evidence/TC-03/error.json и catalog.json; tests.xml.',
        ]),
        ('AT-02. Модель должна улучшать baseline', [
            'REQ-06 / Must: личная модель прохождения используется только при Brier score строго меньше простой оценки по прошлым играм. Иначе остается предварительный расчет.',
            'Вход: синтетический игрок QA_Control, 60 попыток на 12 картах, 30 проходов и 30 провалов. Целевая карта 899999 не сыграна. В трех независимых условиях задаются ошибки модели/baseline: 0.12/0.20, 0.28/0.20 и 0.20/0.20.',
            'AC-06: только 0.12 < 0.20 дает trained и источник validated_personal_logistic. Оба остальных случая дают rejected и предварительный источник. Равенство не считается улучшением.',
            'Шаги: 1. Подготовить тестовую историю. 2. Запустить обучение с первым условием и запросить прогноз. 3. Повторить в чистой базе со вторым условием. 4. Повторить с равными ошибками. 5. Сравнить статус и выбранный источник во всех трех ответах.',
            'Факт: 0.12/0.20 -> trained; 0.28/0.20 -> rejected; 0.20/0.20 -> rejected. Выбранные источники соответствуют статусу.',
            'Итог автоматической проверки: Passed. Покрытие: TC-02, TC-04, TC-05. Замечание: ошибки Brier заданы тестом для проверки правила подключения. Это не замер качества реального игрока.',
            'Evidence: evidence/TC-02, TC-04, TC-05: conditions.json, training.json и prediction.json; tests.xml.',
        ]),
        ('AT-03. Персональный прогноз и нехватка истории', [
            'REQ-05 / Must: для выбранной карты показывается вероятность прохождения. В ТЗ вероятность 0..1; интерфейс и API показывают эквивалент в процентах 0..100.',
            'Вход: QA_Control с достаточной тестовой историей из AT-02 и карта 899999; отдельно QA_TwoResults с ровно двумя результатами и новая карта 600002.',
            'AC-05: при достаточной истории pass_probability - число 0..100 и источник прогноза указан. При двух результатах персональная вероятность неизвестна, observations=2. Не выдавать отсутствие данных за нулевой шанс.',
            'Шаги: 1. Запросить анализ новой карты для QA_Control. 2. Проверить тип, диапазон числа и источник. 3. В другой тестовой базе сохранить два результата QA_TwoResults. 4. Запросить анализ новой карты. 5. Проверить неизвестный pass и число наблюдений.',
            f"Факт: QA_Control получил pass {prediction['pass_probability']}% (вероятность {prediction['pass_probability']/100:.3f}), источник {prediction['model']}. При двух результатах pass_probability=null, observations={insufficient['observations']}; accuracy также не выдумана.",
            'Итог автоматической проверки: Passed. Покрытие: TC-02 и TC-06. Это проверка поведения приложения, не доказательство калибровки вероятности.',
            'Evidence: evidence/TC-02/prediction.json; evidence/TC-06/conditions.json и analysis.json; tests.xml.',
        ]),
        ('Сводка результатов', (['ID', 'REQ / приоритет', 'Итог', 'Замечание'],[
            ['AT-01', 'REQ-02 / Must', 'Passed', 'Оба формата, повтор и неверный файл проверены'],
            ['AT-02', 'REQ-06 / Must', 'Passed', 'Контролируемые Brier: лучше / хуже / равно'],
            ['AT-03', 'REQ-05 / Must', 'Passed', 'Число в диапазоне и неизвестный прогноз при двух результатах'],
        ])),
        ('Что открыть и как повторить', [
            'Открыть этот PDF и папку evidence. Показать одну строку training.json с trained/rejected и соответствующий prediction.json. Для карты показать два upload.json и анализ.',
            'Команда повторения из корня проекта: python -m pytest tests/test_practice_cases.py -q. Она выполняет шесть автоматических проверок на изолированных базах. Базы игрока не затрагиваются.',
        ]),
        ('Решение по приемке после показа', 'Заказчик: Ляхов Я.П. Учебную проверку проводит преподаватель. Решение: принять / с замечаниями / доработать ______. Дата и версия ______. Замечания, ответственный и срок ______. Тесты уже выполнены; подпись и решение заказчика не заполняются за него.'),
    ]
    layout.render('Приемочные проверки - pulse!', 'Три приемочные проверки pulse!',sections)
    speech='''У нас три Must-требования: загрузка карты, правильный выбор модели и персональный прогноз.

Первое, REQ-02. Загружаем одну карту как .osu и как .osz. В обоих случаях совпадают ID и число объектов, повтор не создает копию. Поврежденный файл отклоняется и не меняет каталог. Проверка Passed.

Второе, REQ-06. Проверяем три условия: модель лучше baseline, хуже и ровно равна ему. Подключается только модель с меньшей ошибкой. Во втором и третьем случае она отклоняется. Проверка Passed. Здесь ошибки заданы специально, чтобы проверить правило, это не реальные метрики игрока.

Третье, REQ-05. При достаточной тестовой истории получаем численную вероятность в допустимом диапазоне и видим ее источник. При двух результатах прогноз неизвестен, а не равен нулю. Проверка Passed.

Шесть связанных автотестов только что повторили, все прошли. В PDF есть шаги и фактические ответы, рядом сохранены JSON и журнал pytest. Принимает результат заказчик Ляхов Ярослав; учебные требования проверяет преподаватель. Решение о приемке фиксируем после показа.
'''
    (OUT/'Что говорить.txt').write_text(speech,encoding='utf-8')
    # A fixed portable input lets the reviewer repeat AT-01 without a download.
    inputs=OUT/'inputs';inputs.mkdir(exist_ok=True)
    shutil.copy2(ROOT/'deliverables/test-kit/inputs/valid-map.osu',inputs/'valid-map.osu')
    with zipfile.ZipFile(inputs/'valid-map.osz','w',zipfile.ZIP_DEFLATED) as z:z.write(inputs/'valid-map.osu','valid-map.osu')
    (inputs/'broken-map.osu').write_text('This is not an osu beatmap.\n',encoding='utf-8')
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,OUT.name+'/'+p.relative_to(OUT).as_posix())
    print(OUT)


if __name__=='__main__':build()
