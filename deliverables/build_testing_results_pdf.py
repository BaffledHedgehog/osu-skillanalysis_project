"""Readable, evidence-backed results of the six course test cases."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, PageBreak, KeepTogether)

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'deliverables/testing-debt-2026-11-07'
NAME='06 - Подробные результаты тестов.pdf'
PURPLE=colors.HexColor('#6B32B9')
INK=colors.HexColor('#20283B')
MUTED=colors.HexColor('#566177')
GREEN=colors.HexColor('#14705A')


def build_pdf(folder=DEFAULT):
    folder=Path(folder)
    suites=ET.parse(folder/'practice.xml').getroot().findall('testsuite')
    count={k:sum(int(s.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}
    assert count=={'tests':6,'failures':0,'errors':0,'skipped':0},count
    for name,filename in [('Arial','arial.ttf'),('ArialBold','arialbd.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/filename)))
    normal=ParagraphStyle('Body',fontName='Arial',fontSize=10.2,leading=14.2,textColor=INK,spaceAfter=7)
    small=ParagraphStyle('Small',parent=normal,fontSize=8.8,leading=12,textColor=MUTED)
    h1=ParagraphStyle('Title',parent=normal,fontName='ArialBold',fontSize=23,leading=28,spaceAfter=14)
    h2=ParagraphStyle('Heading',parent=normal,fontName='ArialBold',fontSize=11.2,leading=15,textColor=PURPLE,spaceBefore=8,spaceAfter=5)
    table_text=ParagraphStyle('Table',parent=normal,fontSize=9,leading=12,spaceAfter=0)
    table_head=ParagraphStyle('TableHeader',parent=table_text,fontName='ArialBold',textColor=colors.white)
    status=ParagraphStyle('Status',parent=normal,fontName='ArialBold',fontSize=12,leading=17,textColor=GREEN)
    story=[]
    def p(text,style=normal):return Paragraph(escape(str(text)).replace('\n','<br/>'),style)
    def block(title,body):
        lines=body if isinstance(body,list) else [body]
        return KeepTogether([p(title,h2)]+[p(line) for line in lines])
    def load(case,name):return json.loads((folder/'evidence'/case/name).read_text(encoding='utf-8'))
    first=load('TC-01','valid-map.osu-analysis.json')
    error=load('TC-03','error.json')
    catalog=load('TC-03','catalog.json')
    cases=[
        dict(id='TC-01',title='Загрузка карты .osu и .osz',req='REQ-02 / Must',kind='Успешный',
             purpose='Проверить, что приложение принимает обычный файл карты и архив с той же картой, правильно читает содержимое и не создает повтор в каталоге.',
             data=f"Карта {first['id']}: {first['title']} [{first['version']}]. В исходнике {first['object_count']} объектов. Подготовлены .osu и .osz с этой картой. Профиль игрока для проверки импорта не нужен.",
             steps=['1. Загрузили файл .osu через API приложения.',
                    '2. Открыли анализ и сравнили ID, название, число объектов и признаки с исходным файлом.',
                    '3. Загрузили .osz с той же картой, повторили сравнение и проверили каталог.'],
             expected='Оба формата принимаются. Данные карты совпадают с исходником, нагрузка рассчитывается. Повторная загрузка той же карты не добавляет еще один ID.',
             actual=f"Оба запроса вернули HTTP 200. ID: {first['id']}. Объектов: {first['object_count']}. Звезды: {first['stars']}. Aim: {first['aim']}. Speed: {first['speed']}. После двух загрузок в каталоге остался один ID этой карты.",
             user='Пользователь может добавить карту любым из двух форматов и открыть ее анализ. Ему доступны название, сложность и нагрузка по навыкам. Повторная загрузка не засоряет каталог копиями.',
             evidence='Ответы API двух загрузок, результаты анализа и проверка числа карт.'),
        dict(id='TC-02',title='Модель лучше базового метода',req='REQ-06 / Must',kind='Успешный',
             purpose='Проверить правило подключения ML: модель прохождения должна использоваться, когда ее ошибка строго меньше ошибки базового метода.',
             data='Тестовый игрок QA_Control: 60 попыток на 12 картах, 30 проходов и 30 провалов. Для проверки оставлены 3 карты и 15 попыток. Brier модели задан равным 0.12, базового метода - 0.20. Целевая карта 899999 отсутствует в истории игрока.',
             steps=['1. Подготовили отдельную базу и синтетическую историю игрока.',
                    '2. Задали контрольные значения Brier и запустили обучение через API.',
                    '3. Проверили решение о подключении и запросили прогноз целевой карты.',
                    '4. Убедились, что карты обучения и проверки не пересекаются.'],
             expected='Так как 0.12 < 0.20, модель получает статус trained. Прогноз прохождения использует личную логистическую модель.',
             actual='',user='',evidence='Контрольные условия, отчет обучения и ответ API прогноза. Числа Brier заданы тестом; это не замер качества на реальном игроке.'),
        dict(id='TC-03',title='Поврежденный файл карты',req='REQ-02 / Must',kind='Негативный',
             purpose='Проверить, что неверный файл не принимается за карту, не меняет каталог и не нарушает работу приложения.',
             data='В отдельной базе уже есть карта 100. Подготовлен обычный текст без структуры карты, но с расширением .osu. До загрузки сохранен список ID каталога.',
             steps=['1. Сохранили список карт до загрузки.',
                    '2. Отправили поврежденный .osu в загрузчик.',
                    '3. Проверили ошибку, повторно прочитали каталог и вызвали проверку работоспособности сервера.'],
             expected='Приложение объясняет, что файл неверный. Новая карта не появляется, существующие данные сохраняются, сервер продолжает отвечать.',
             actual=f"Загрузка вернула HTTP {error['http_status']}. Текст ошибки: \"{error['detail']}\". Каталог до: {', '.join(catalog['before'])}; после: {', '.join(catalog['after'])}. Списки совпали. Проверка работоспособности вернула HTTP 200.",
             user='Пользователь получает понятное сообщение "Это не файл карты .osu." и может выбрать другой файл. Его библиотека остается на месте; перезапуск приложения не требуется.',
             evidence='Ответ с ошибкой, списки каталога до и после, успешный ответ проверки работоспособности.'),
        dict(id='TC-04',title='Модель хуже базового метода',req='REQ-06 / Must',kind='Негативный',
             purpose='Проверить, что модель с большей ошибкой не подключается как улучшение. Приложение должно сохранить доступный предварительный расчет.',
             data='Тестовый игрок QA_Control: те же 60 попыток на 12 картах. Brier модели задан равным 0.28, базового метода - 0.20. Для запроса используется новая целевая карта 899999.',
             steps=['1. Подготовили синтетическую историю в отдельной базе.',
                    '2. Задали контрольные ошибки 0.28 и 0.20, запустили обучение.',
                    '3. Проверили статус модели и источник результата анализа карты.'],
             expected='При 0.28 > 0.20 модель получает статус rejected. Для прохождения используется предварительный расчет, а не эта ML-модель.',
             actual='',user='',evidence='Решение об отклонении модели и ответ API с источником предварительного расчета.'),
        dict(id='TC-05',title='Ошибка модели равна базовой',req='REQ-06 / Must',kind='Граничный',
             purpose='Проверить точную границу условия. Равенство ошибок не должно считаться улучшением модели.',
             data='Тестовый игрок QA_Control: 60 попыток на 12 картах. Обе ошибки Brier явно заданы равными 0.20. Используется новая карта 899999, которой нет в истории.',
             steps=['1. Подготовили изолированную историю с проходами и провалами.',
                    '2. Задали одинаковые значения Brier: 0.20 и 0.20.',
                    '3. Запустили обучение и проверили решение при равенстве.',
                    '4. Запросили анализ карты и проверили, какой расчет выбран.'],
             expected='Модель получает статус rejected. Условие строгое: нужно меньше, а не меньше или равно. Приложение использует предварительный расчет.',
             actual='',user='',evidence='Одинаковые значения метрик, статус rejected и источник прогноза из API. Отклонение модели в этом сценарии - правильная работа, поэтому итог теста Pass.'),
        dict(id='TC-06',title='У игрока недостаточно результатов',req='REQ-05, REQ-09 / Must',kind='Негативный',
             purpose='Проверить, что приложение не выдает выдуманный персональный прогноз, пока минимальная история не собрана.',
             data='Тестовый игрок QA_TwoResults имеет ровно 2 полных результата на карте 600001. Для анализа добавлена карта 600002. Минимум для первого общего прогноза - 3 результата; это отдельное условие от минимального числа карт для шкал навыков.',
             steps=['1. Создали тестового игрока с двумя результатами.',
                    '2. Запросили анализ карты 600002 для этого игрока.',
                    '3. Проверили прогноз, объяснение нехватки истории и доступность сервера.'],
             expected='Карта анализируется, но шанс прохождения и ожидаемая accuracy отсутствуют. Пользователь получает объяснение, сколько данных нужно для прогноза.',
             actual='',user='',evidence='Ответ анализа с двумя наблюдениями, пустыми числовыми прогнозами и сообщением о минимуме из трех результатов.'),
    ]
    for case in cases:
        cid=case['id']
        if cid in ['TC-02','TC-04','TC-05']:
            train=load(cid,'training.json');prediction=load(cid,'analysis.json')['prediction']
            chosen='личная логистическая модель' if cid=='TC-02' else 'предварительный расчет по истории'
            case['actual']=(f"Brier модели: {train['metrics']['brier']:.2f}; базового метода: {train['metrics']['baseline_brier']:.2f}. "
                            f"Статус модели: {train['status']}. Выбран {chosen}. "
                            f"API вернул шанс прохождения {prediction['pass_probability']}% и ожидаемую accuracy {prediction['expected_accuracy']}%. "
                            f"Диапазон accuracy: {prediction['accuracy_range'][0]}-{prediction['accuracy_range'][1]}%.")
            if cid=='TC-02':
                case['user']='Для тестового игрока рассчитан шанс прохождения 57.3% с использованием ML. Источник расчета указан отдельно от оценки accuracy. В этом тесте accuracy 94.4% получена предварительным методом; обучение accuracy - другая задача.'
            else:
                case['user']='Игрок получает предварительный шанс прохождения 65.8% и ожидаемую accuracy 94.4%. Проверяемая ML-модель не выдается за источник этого прогноза. Приложение продолжает выдавать доступную оценку по истории.'
        if cid=='TC-06':
            prediction=load(cid,'analysis.json')['prediction']
            case['actual']=(f"HTTP 200. Число наблюдений: {prediction['observations']}. "
                            'Шанс прохождения, expected_accuracy и accuracy_range: null, то есть значение отсутствует. '
                            f"model: {prediction['model']}. Сообщение: \"{prediction['message']}\". Сервер отвечает.")
            case['user']='Вместо процентов игрок видит сообщение о двух результатах из трех необходимых. Можно изучать нагрузку карты и добавить историю. Отсутствие прогноза не показывается как нулевой шанс прохождения.'

    story += [p('Результаты тестирования pulse!',h1),
              p('Подробный отчет по 6 выполненным кейсам / 07.11.2026',small),Spacer(1,12),
              p('6 из 6 проверок пройдены',status),
              p('Для каждого кейса описаны исходные данные, действия, ожидаемый ответ и фактический результат. Поле "Что получает пользователь" объясняет результат простыми словами.'),
              p('Это отдельный отчет по тестированию MVP. Проверки выполнены автоматически на изолированных базах. В TC-02, TC-04, TC-05 и TC-06 используются тестовые игроки, а не история реального пользователя.'),
              p('Brier - ошибка вероятностей: меньше означает лучше. Значения 0.12, 0.20 и 0.28 в этих кейсах заданы специально для проверки правила выбора модели. Проценты прогноза ниже - фактические ответы API на контрольных данных, а не результат сыгранной человеком карты.'),Spacer(1,10)]
    rows=[[p('Кейс',table_head),p('Проверка',table_head),p('Итог',table_head)]]
    rows += [[p(c['id'],table_text),p(c['title'],table_text),p('Pass',table_text)] for c in cases]
    summary=Table(rows,colWidths=[1.8*cm,12.1*cm,2.7*cm],hAlign='LEFT')
    summary.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PURPLE),
                                ('BACKGROUND',(0,1),(-1,-1),colors.HexColor('#F6F3FB')),
                                ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#DDD6E8')),
                                ('VALIGN',(0,0),(-1,-1),'TOP'),
                                ('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),
                                ('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9)]))
    story += [summary,Spacer(1,14),p('Состав: 2 успешных, 3 негативных, 1 граничный. Итог прогона: 6 passed, 0 failed, 0 blocked.',small),
              p('Отрицательные входные условия не означают провал теста. Если приложение правильно отклоняет неверный файл или неподходящую модель, проверка считается успешной.',small)]
    for case in cases:
        story += [PageBreak(),p(case['id']+'. '+case['title'],h1),
                  p(case['req']+' | '+case['kind']+' | Проверено 07.11.2026',small),
                  block('В чем заключался тест',case['purpose']),
                  block('Что подготовили',case['data']),
                  block('Что сделали',case['steps']),
                  block('Что ожидали',case['expected']),
                  block('Что получилось фактически',case['actual']),
                  block('Что получает пользователь',case['user']),
                  Spacer(1,5),p('Результат теста: Pass',status),p(case['evidence'],small)]
    def decorate(canvas,doc):
        w,h=A4
        canvas.setFillColor(PURPLE);canvas.rect(0,h-9,w,9,stroke=0,fill=1)
        canvas.setFont('Arial',8);canvas.setFillColor(MUTED)
        canvas.drawString(42,24,'pulse! / Выполненные проверки MVP / 07.11.2026')
        canvas.drawRightString(w-42,24,str(doc.page))
    target=folder/NAME
    doc=SimpleDocTemplate(str(target),pagesize=A4,leftMargin=42,rightMargin=42,
                          topMargin=36,bottomMargin=42,title='pulse! - подробные результаты шести тестов',
                          author='Команда проекта pulse!')
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    return target


if __name__=='__main__':print(build_pdf())
