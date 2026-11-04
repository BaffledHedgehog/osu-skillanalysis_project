"""Fill the course documents from recorded evidence, keeping failures visible."""
import csv
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

from build_practice_tests import build as build_cases
from build_test_plan import build as build_plan

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT.parent / '.data/test-evidence/2026-11-04'
KIT = ROOT / 'test-kit'


def load(relative):
    return json.loads((EVIDENCE / relative).read_text(encoding='utf-8'))


def build():
    practice = ET.parse(EVIDENCE / 'practice.xml')
    assert len(practice.findall('.//testcase')) == 5
    assert not practice.findall('.//failure') and not practice.findall('.//error')
    suite = ET.parse(EVIDENCE / 'full-suite.xml')
    total = len(suite.findall('.//testcase'))
    skipped = len(suite.findall('.//skipped'))
    assert total == 107 and skipped == 1 and not suite.findall('.//failure')
    native = ET.parse(EVIDENCE / 'native-audio.xml')
    assert len(native.findall('.//failure')) == 1
    browser = load('browser-reference/results.json')
    assert not browser['errors'] and browser['cases']['TC-03']['catalog_unchanged']
    controlled = load('browser-reference/controlled-results.json')
    assert [x['uses_ml'] for x in controlled] == [True, False, False]
    live = load('live-4610937/final-result.json')
    cases = {
        'TC-01': {'actual': 'Оба файла приняты через API и браузер: HTTP 200. Одна карта 4864861, 126 объектов, 5.57 звезд, aim 3.04, speed 2.26. Ошибок JavaScript нет.', 'evidence': 'automated/TC-01/*.json; browser-reference/results.json; TC-01-osu.png; TC-01-osz.png'},
        'TC-02': {'actual': 'Контрольные Brier 0.12 и 0.20: модель trained, API использует validated_personal_logistic. Карты обучения и проверки не пересекаются. Браузер показывает ML: ШАНС PASS.', 'evidence': 'automated/TC-02/*.json; practice.xml; browser-reference/TC-02-controlled.png'},
        'TC-03': {'actual': 'API и браузер отклонили broken-map.osu с HTTP 400. Ошибка видна в интерфейсе. Список ID карт до и после совпал; сервер продолжил отвечать.', 'evidence': 'automated/TC-03/*.json; browser-reference/results.json; TC-03-error.png'},
        'TC-04': {'actual': 'Контрольные Brier 0.28 и 0.20: rejected. API и браузер используют предварительный расчет и не называют его проверенным ML.', 'evidence': 'automated/TC-04/*.json; practice.xml; browser-reference/TC-04-controlled.png'},
        'TC-05': {'actual': 'Контрольные Brier равны: 0.20. Модель rejected, прогноз предварительный. Равенство не приводит к trained. В браузере показан предварительный шанс pass.', 'evidence': 'automated/TC-05/*.json; practice.xml; browser-reference/TC-05-controlled.png'},
    }
    for item in cases.values():
        item['status'] = 'Pass'
    checks = [
        {'name': 'TC-01..05', 'expected': 'Два успешных, два негативных и один граничный кейс', 'actual': '5 Pass. API и интерфейс проверены автоматически. Brier в TC-02/04/05 задан в тесте; интерфейс получает сохраненные контрольные ответы API.', 'status': 'Pass'},
        {'name': 'Полный набор тестов', 'expected': 'Основные тесты проходят; отдельный запуск проверки native host', 'actual': 'Основной запуск: 106 passed, 1 skipped. Затем пропущенный opt-in тест .NET запущен отдельно: Fail, таймаут 30 секунд.', 'status': 'Fail: NATIVE-01'},
    ]
    for check in load('acceptance/checks.json')['checks']:
        checks.append({**check, 'actual': json.dumps(check['actual'], ensure_ascii=False, indent=2)})
    checks.append({'name': 'Новая попытка: карта 4610937', 'expected': 'Сохранить прогноз до сообщения результата и сравнить с фактом', 'actual': f"Прогноз pass 92.1%, accuracy 82.3%, диапазон 72.8-91.8%. Игрок сообщил pass с первого раза и 94.01%. Ошибка accuracy {live['accuracy_error_pp']} п.п.; факт вне диапазона. Brier 0.006241 против 0.004699 у базового прогноза. Ошибка базового accuracy 9.6677 п.п.", 'status': 'Проверено; промах accuracy, QUALITY-01'})
    final_suite = ET.parse(EVIDENCE / 'final-suite.xml')
    final_count = len(final_suite.findall('.//testcase'))
    final_skip = len(final_suite.findall('.//skipped'))
    assert not final_suite.findall('.//failure') and not final_suite.findall('.//error')
    checks.append({'name': 'Финальный прогон после изменений', 'expected': 'Все основные тесты проходят, новый импорт и рекомендации проверены',
                   'actual': f'{final_count-final_skip} passed, {final_skip} skipped. Пропущен опциональный native-тест, его отдельный Fail сохранен выше. Проверены полный исходный результат, четыре списка, NF, повторы, резервный источник, все клиенты и исключения карт.', 'status': 'Pass; native Fail не устранен'})
    updated = load('updated-browser/results.json')
    assert not updated['js_errors'] and updated['no_known_or_owned_recommendations']
    stars = [r['stars'] for r in updated['recommended_maps']]
    checks.append({'name': 'Живой сбор одной кнопкой и новые карты', 'expected': 'История дополнена, настройки свернуты, знакомые и слишком простые карты исключены',
                   'actual': updated['dataset_status'] + f" Подбор: {len(stars)} карт, {min(stars)}-{max(stars)} звезд. Исключено {updated['excluded_maps']} известных карт. Нижняя сложность {updated['min_stars']} звезд. Ошибок JavaScript нет. Локальная проверка на порту 8011, копия базы.", 'status': 'Pass'})
    result = {
        'display_date': '04.11.2026',
        'version': 'commit 29ab165 и рабочие изменения от 04.11.2026, отдельная копия базы, порт 8010',
        'summary': f'Проверки выполнены автоматически через pytest, API и браузер. Пять учебных кейсов пройдены. Финальный набор: {final_count-final_skip} passed, {final_skip} skipped. Дополнительно проверены моды, две контрольные истории, пустая история, реальное обучение, живой импорт и повтор без дублей, сбор одной кнопкой и новые карты. Отдельно сохранены неудачный запуск .NET и промах accuracy на новой попытке игрока. Приемка другим участником по README не выполнялась.',
        'completion': 'Начало выполнено: стенд доступен, карта и профиль загружены. TC-01..05 прошли. Полная приемка MVP не объявлена: запуск .NET завершился Fail; на новой попытке accuracy отличается от факта на 11.71 п.п. Нужна проверка качества на серии новых попыток и запуск по README другим участником.',
        'cases': cases, 'checks': checks,
        'issues': [
            {'id': 'NATIVE-01', 'title': 'Не завершилась проверка запуска .NET host', 'details': 'Опциональный native-тест остановлен по таймауту 30 секунд. Проверка выполнялась без реплея и без воспроизведения звука. Защита запретила дальнейший запуск. Причина не установлена; функции HP по этому движку не подтверждены.'},
            {'id': 'QUALITY-01', 'title': 'Занижен прогноз accuracy на карте 4610937', 'details': '82.3% вместо сообщенных 94.01%, ошибка 11.71 п.п., факт вне показанного диапазона. На этой попытке простой прогноз также ошибся, но меньше: 9.6677 п.п. Одна попытка не доказывает качество модели в целом. Нужно проверить признаки, состав истории и серию новых игр.'},
            {'id': 'PERF-01', 'title': 'Один запрос анализа не завершился за 40 секунд', 'details': 'Таймаут наблюдался во время импорта. Повтор после импорта вернул HTTP 200 за 0.1 секунды. Причина и повторяемость не установлены; нужна отдельная проверка одновременных запросов.'},
        ],
        'live': {
            'description': 'BaffledHedgehog, Yoake ni Sayonara [Farewell at Dawn], карта 4610937, 6.18 звезд. Прогноз сохранен 04.11.2026 в 20:45:35 по Москве до сообщения результата. Моды NM приняты из запроса прогноза, игрок отдельно их не подтвердил. Результат взят из сообщения игрока, а не из проверенного реплея. Прогноз: pass 92.1%, accuracy 82.3%, диапазон 72.8-91.8%.',
            'actual': 'Pass с первой попытки, accuracy 94.01%. Ошибка accuracy 11.71 п.п.; результат выше верхней границы на 2.21 п.п. Brier 0.006241. Простой базовый прогноз на этой попытке оказался лучше по обоим показателям.',
            'status': 'Результат записан. Accuracy требует улучшения. Новая попытка не включалась в обучение до фиксации сравнения.',
        },
    }
    KIT.mkdir(exist_ok=True)
    (KIT / 'test-results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    backup = EVIDENCE / 'original-documents'
    backup.mkdir(exist_ok=True)
    for name in ['Тест-план MVP - pulse!.docx', 'Практика 1 - Тест-кейсы MVP - pulse!.docx']:
        if not (backup / name).exists():
            shutil.copy2(ROOT / name, backup / name)
    build_cases(); build_plan()
    with (KIT / 'Журнал прогона.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['TC-ID', 'REQ-ID', 'Способ проверки', 'Фактический результат', 'Статус', 'Evidence'])
        for tc, item in cases.items():
            writer.writerow([tc, 'REQ-02' if tc in ('TC-01', 'TC-03') else 'REQ-06', 'Автотест API и автоматический браузер; без ручной приемки', item['actual'], item['status'], item['evidence']])
        for item in checks[1:]:
            writer.writerow([item['name'], '', 'Автоматически; результат новой игры сообщен игроком', item['actual'], item['status'], 'См. тест-план и архив'])
    archive = KIT / 'Доказательства тестов - pulse!.zip'
    with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
        for relative in ['practice.xml', 'pytest.txt', 'full-suite.xml', 'full-suite.txt', 'native-audio.xml', 'native-audio.txt', 'final-suite.xml', 'final-suite.txt']:
            z.write(EVIDENCE / relative, relative)
        for directory in ['automated', 'browser-reference', 'acceptance', 'live-4610937', 'updated-browser']:
            for p in (EVIDENCE / directory).rglob('*'):
                if p.is_file() and p.suffix in ('.json', '.png'):
                    z.write(p, p.relative_to(EVIDENCE))
        z.write(KIT / 'test-results.json', 'test-results.json')
    print(f'Filled 2 DOCX and journal; evidence archive {archive.stat().st_size} bytes')


if __name__ == '__main__':
    build()
