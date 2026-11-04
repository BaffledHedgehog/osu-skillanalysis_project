"""Make a readable recruitment workbook from verified public leaderboard rows."""
import csv
import json
from pathlib import Path

import xlsxwriter

ROOT = Path(__file__).resolve().parent


def build():
    with (ROOT / 'dataset-candidates.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f, delimiter=';'))
    summary = json.loads((ROOT / 'dataset-candidates-summary.json').read_text(encoding='utf-8'))
    assert len({r['user_id'] for r in rows}) == len(rows)
    for row in rows:
        low, high = map(int, row['pp_bin'].split('-'))
        assert low <= float(row['pp']) < high
        assert row['profile_url'] == f"https://osu.ppy.sh/users/{row['user_id']}"
        assert row['consent'] in ('', 'owner_confirmed')
    output = ROOT / 'Кандидаты в датасет - pulse!.xlsx'
    book = xlsxwriter.Workbook(output, {'strings_to_formulas': False, 'strings_to_urls': False})
    header = book.add_format({'bold': True, 'bg_color': '#6025A5', 'font_color': 'white', 'text_wrap': True})
    wrap = book.add_format({'text_wrap': True, 'valign': 'top'})
    pp_format = book.add_format({'num_format': '0.0'})
    guide = book.add_worksheet('Как собирать')
    guide.set_column('A:A', 27); guide.set_column('B:B', 110)
    notes = [
        ('Цель пилота', '80-120 согласившихся игроков. Это рабочий ориентир для первого обучения, а не расчет необходимого размера выборки.'),
        ('Список', f"{summary['primary']} основных и {summary['reserve']} запасных кандидата. " + (f"Владелец проекта сообщил о согласии всех {summary['consent_report']['players']} игроков 04.11.2026. Индивидуальные подтверждения в проект не загружались." if summary.get('consent_report') else 'Согласий пока нет.')),
        ('Диапазоны pp', 'Шаг 2000 pp, до 6 основных и 6 запасных в каждом диапазоне. Выше 24000 pp кандидатов мало. Не нужно задерживать пилот ради согласия топ-игроков.'),
        ('Кого добавлять', 'В первую очередь закрывать пустые диапазоны и разные стили игры: aim, speed, tech, reading, HD и другие. Одинаковые pp не означают одинаковые навыки.'),
        ('Источник', 'Публичные глобальный и национальные рейтинги через Hinamizawa. Дата получения: 04.11.2026. Зеркало кеширует данные, поэтому pp могут отличаться от текущих на сайте.'),
        ('Ограничение выборки', 'Это список для приглашения, не случайная выборка всех игроков. Использованы первые страницы 35 стран и глобального рейтинга. Внизу pp много LI и MT. Не проверялись последняя активность и доступность реплеев.'),
        ('Что просить', 'Согласие на использование игровых результатов для обучения и оценки учебной модели. Отдельно согласовать локальные scores.db и реплеи, срок хранения и возможность публикации.'),
        ('Что полезно для модели', 'Обычные попытки с pass и fail, дата, карта, моды, accuracy, hit statistics. Одних best недостаточно. Для начала целиться в 100-300 разных попыток на игрока, если доступны.'),
        ('NoFail', 'Хранить отдельно. Сам факт завершения с NF не доказывает прохождение без NF. HP-метку использовать только при проверенном расчете.'),
        ('Безопасная передача', 'Не просить пароли, cookie и OAuth-токены. Не собирать другие файлы клиента. Реплеи и исходные базы не публиковать без отдельного согласия.'),
        ('Согласия', 'На листах Основные и Запасные заполнить согласие, дату, объем и ссылку на подтверждение. Публичные данные игрока сами по себе не являются его согласием.'),
        ('Проверка качества', 'Разделять обучение и проверку по игрокам, картам и времени. Все отражения одной карты оставлять в одной части выборки. Сравнивать с простым расчетом и смотреть метрики отдельно по pp и модам.'),
        ('Как понять, что хватит', 'Построить кривую обучения на 20, 40, 80 и следующих количествах игроков. Если ошибка на отложенных игроках продолжает заметно падать, собирать больше. Считать также число игроков и неудач в каждом диапазоне.'),
        ('Текст на русском', 'Привет! Я делаю учебный проект pulse!: анализ навыков osu!standard и прогноз pass/accuracy. Можно использовать твои игровые результаты для обучения и проверки модели? Нужны обычные попытки, в том числе fail, с картой, модами, датой и accuracy. Если удобно, отдельно обсудим экспорт scores.db и выбранных реплеев. Пароли и cookie не нужны. Исходные файлы не буду публиковать без отдельного согласия. До передачи согласуем, какие данные берем и на какой срок.'),
        ('Message in English', 'Hi! I am building pulse!, a student project for osu!standard skill analysis and pass/accuracy prediction. May I use your gameplay results to train and evaluate the model? Regular attempts, including fails, with beatmap, mods, date and accuracy are useful. If you are comfortable, we can separately discuss a scores.db export and selected replays. I do not need passwords or cookies. I will not publish your source files without separate permission. Before you share anything, we can agree on the data scope and retention period.'),
    ]
    for i, (label, value) in enumerate(notes):
        guide.write(i, 0, label, header); guide.write(i, 1, value, wrap)
        guide.set_row(i, 48 if len(value) < 300 else 92)
    columns = ['Диапазон pp', 'Ник', 'pp', 'Страна', 'ID', 'Профиль', 'Число игр', 'Место', 'Получено', 'Источник', 'Согласие', 'Дата согласия', 'На какие данные', 'Подтверждение', 'Комментарий']
    for key, name in [('primary', 'Основные'), ('reserve', 'Запасные')]:
        sheet = book.add_worksheet(name)
        selected = [r for r in rows if r['list'] == key]
        sheet.add_table(0, 0, len(selected), len(columns)-1, {'columns': [{'header': c} for c in columns], 'style': 'Table Style Medium 4'})
        for i, r in enumerate(selected, 1):
            values = [r['pp_bin'], r['username'], float(r['pp']), r['country'], int(r['user_id']), r['profile_url'], int(r['play_count']), int(r['global_rank'] or 0), r['verified_at'][:19], r['source_url'], '', '', '', '', '']
            if r['consent'] == 'owner_confirmed':
                values[10:] = ['Да', '04.11.2026 (сообщение владельца)', 'Публичные игровые результаты: обучение и проверка', 'Сообщение владельца проекта в чате', 'Дата индивидуального согласия не указана. Локальные файлы, реплеи и публикация отдельно не согласованы.']
            for j, value in enumerate(values):
                sheet.write(i, j, value, pp_format if j == 2 else None)
            sheet.write_url(i, 5, r['profile_url'], string=r['profile_url'])
        sheet.freeze_panes(1, 2)
        sheet.set_column('A:A', 17); sheet.set_column('B:B', 24); sheet.set_column('C:E', 12)
        sheet.set_column('F:F', 39); sheet.set_column('G:I', 21); sheet.set_column('J:J', 32)
        sheet.set_column('K:O', 24)
        sheet.data_validation(1, 10, len(selected), 10, {'validate': 'list', 'source': ['Не писал', 'Ожидаю ответ', 'Да', 'Нет', 'Отозвано']})
    book.close()
    print(f'{output.name}: {len(rows)} unique candidates; bins and profile IDs checked')


if __name__ == '__main__':
    build()
