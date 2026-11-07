"""Prepare local upload copies after a read-only inventory of the team Drive.

Does not upload files, sign acceptance, or rewrite historical baseline copies.
"""
from pathlib import Path
from datetime import date
import json
import shutil
import zipfile

from docx import Document

import build_course_pack as course
from build_mvp_canvas import build_canvas

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/drive-upload-2026-11-07'
SOURCE=ROOT/'deliverables/course-all-weeks-2026-11-07'
URL='https://drive.google.com/drive/folders/1GnNYKprJV9hBir-iRj0GLmCaARXnuErT'


def build():
    replace=OUT/'01 - Заменить старые';add=OUT/'02 - Добавить'
    replace.mkdir(parents=True,exist_ok=True);add.mkdir(exist_ok=True)
    shutil.copy2(SOURCE/'14 - Актуальный паспорт.docx',replace/'Паспорт проекта - pulse!.docx')
    shutil.copy2(SOURCE/'14 - Актуальный паспорт.pdf',replace/'Паспорт проекта - pulse!.pdf')
    # Keep each original REQ ID, source, MoSCoW and acceptance check.
    registry=Document(ROOT/'deliverables/Реестр требований - pulse!.docx')
    for p in registry.paragraphs:
        if p.text.startswith('Версия 0.2'):
            p.text='Версия 0.3 от 07.11.2026. Заказчик - Ляхов Я.П. Реестр уточнен к досрочной защите. Источники, MoSCoW и критерии проверки сохранены для каждого требования.'
        if p.text.startswith('После интервью уточнить'):
            p.text='Базовый набор - aim и speed; 16 групп добавляются поэтапно. Независимый будущий пилот и проверка других игроков остаются отдельными задачами. Решение о приемке фиксируется после показа.'
    states={'REQ-01':'Реализовано; доступность зависит от источника','REQ-02':'Проверено: TC-01/03',
            'REQ-03':'Базовые aim/speed проверены; 16 групп поэтапно','REQ-04':'Базовая относительная нагрузка проверена; дополнительные шкалы в предпросмотре',
            'REQ-05':'Прогноз и неизвестная оценка при нехватке данных проверены','REQ-06':'Проверено: TC-02/04/05',
            'REQ-07':'Базовый подбор реализован; полный план тренировок отложен','REQ-08':'Не входит в MVP',
            'REQ-09':'Личная V3; историческая проверка NM, будущий пилот открыт',
            'REQ-10':'Внутренняя проверка есть; независимая будущая серия не завершена',
            'REQ-11':'MAE на истории измерена; будущий независимый пилот не завершен',
            'REQ-12':'Дедупликация и разные попытки проверены','REQ-13':'Фильтры меток проверены',
            'REQ-14':'Реализовано; нативный replay/HP отдельно от базового сценария','REQ-15':'Базовый сценарий без секретов реализован'}
    for table in registry.tables:
        for row in table.rows:
            key=row.cells[0].text
            if key in states and len(row.cells)>=7:row.cells[5].text=states[key]
    registry.save(replace/'Реестр требований - pulse!.docx')
    spec=Document(ROOT/'deliverables/Техническое задание - pulse!.docx')
    for p in spec.paragraphs:
        if '14.11.2026' in p.text:p.text=p.text.replace('14.11.2026','07.11.2026')
        if p.text=='pulse! - черновик для перехода к MVP':p.text='pulse! - учебный MVP к защите'
        if p.text.startswith('Версия 0.2, дата'):
            p.text='Версия 0.3 от 07.11.2026. Заказчик - Ляхов Я.П. Досрочная защита и сдача: 07.11.2026. Решение по приемке оформляется после проверки.'
        if 'реестре требований версии 0.2' in p.text:p.text=p.text.replace('версии 0.2','версии 0.3 от 07.11')
        if p.text.startswith('Остальные границы перечислены'):
            p.text='Базовая приемка включает aim и speed. Полные 16 групп внедряются поэтапно. Актуальные WBS и календарь заканчиваются 07.11. Cookie/OAuth остаются необязательными расширениями.'
    for table in spec.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.startswith('Вход через cookie, пароль'):
                    cell.text='Обязательный пароль или OAuth игрока. Необязательные cookie/OAuth не входят в основной сценарий.'
                if cell.text.startswith('Массовый сбор данных чужих аккаунтов'):
                    cell.text='Сбор без согласия владельцев и обучение pass на best как на полной истории попыток.'
    spec.add_heading('Уточнение состояния на 07.11.2026',level=1)
    spec.add_paragraph('Реализованы импорт, базовые личные шкалы, прогноз и подбор. Accuracy V3 проверена повторно на сохраненной истории одного игрока: NM MAE 3.87 п.п. против V2 4.35. Это не независимый будущий пилот. Экспериментальная поправка по двум свежим результатам включена только на предпросмотре 8012 и не подменяет этот benchmark.')
    spec.add_paragraph('Приемочные проверки: REQ-02 - карта; REQ-06 - выбор модели; REQ-05 - прогноз и нехватка истории. Подробности и факты в пакете приемки. Формальное решение заказчика и преподавателя заранее не записывается.')
    spec.save(replace/'Техническое задание - pulse!.docx')
    old_root=course.ROOT
    course.ROOT=replace
    entries,_=course.schedule();course.build_scope_wbs_decisions(entries)
    course.ROOT=old_root
    shutil.copy2(ROOT/'deliverables/Календарный план - pulse!.pod',replace/'Календарный план - pulse!.pod')
    shutil.copy2(ROOT/'deliverables/Календарный план - pulse!.xlsx',replace/'Календарный план - pulse!.xlsx')
    build_canvas(course.TEAM,date(2026,11,7),replace)
    # Missing weeks: preserve the existing document titles for easy navigation.
    for prefix in ['01 -','02 -','03 -','04 -','05 -','06 -','07 -','08 -','09 -','11 -','15 -']:
        for p in SOURCE.glob(prefix+'*'):
            if p.suffix in ('.pdf','.docx'):shutil.copy2(p,add/p.name)
    shutil.copy2(SOURCE/'Backlog - импорт.csv',add/'Backlog - импорт.csv')
    for name,source in [('Тестирование',ROOT/'deliverables/pulse-testing-debt-2026-11-07'),
                        ('Дополнительные проверки',ROOT/'deliverables/extended-testing-2026-11-07'),
                        ('Приемка',ROOT/'deliverables/acceptance-2026-11-07'),
                        ('Инструкции',ROOT/'docs'),('Защита',ROOT/'deliverables/defense-2026-11-07')]:
        dest=add/name
        shutil.copytree(source,dest,dirs_exist_ok=True)
    inventory=json.loads((ROOT/'.data/drive-audit-2026-11-07/team-index.json').read_text(encoding='utf-8'))
    (OUT/'Drive inventory.json').write_text(json.dumps({'folder_url':URL,'checked_date':'2026-11-07','remote_files':inventory},ensure_ascii=False,indent=2),encoding='utf-8')
    text='''ПРОВЕРКА ПАПКИ КОМАНДЫ НА GOOGLE DRIVE
07.11.2026

Папка команды:
'''+URL+'''

Сейчас на диске 7 файлов:
1. Границы WBS и решения - pulse!.docx (17 октября)
2. Календарный план - pulse!.pod (17 октября)
3. Паспорт проекта - pulse!.docx (17 октября)
4. Реестр требований - pulse!.docx (17 октября)
5. Техническое задание - pulse!.docx (17 октября)
6. MVP Canvas - pulse!.pdf (24 октября)
7. repository.txt (17 октября)

ЧТО ЗАМЕНИТЬ
В папке "01 - Заменить старые" подготовлены актуальные версии первых
шести документов. В старых паспорте, ТЗ и Canvas указан срок 14 ноября.
Новые версии рассчитаны на защиту 7 ноября. Реестр сохраняет отдельные
ID, источники, приоритеты и проверки, состояния требований обновлены.
repository.txt можно оставить, если он ведет на текущий репозиторий.

ЧТО ДОБАВИТЬ
Содержимое "02 - Добавить":
- Стейкхолдеры и 10 вопросов заказчику.
- Риски.
- Backlog, ответственные, зависимости, DoR/DoD и CSV для задач.
- Решения и изменения.
- Данные, baseline и эксперименты.
- Сценарий демонстрации.
- Рабочее обсуждение и форма протокола встречи.
- Тест-план, 6 кейсов, Bug Log, отчет, подробные результаты и evidence.
- Инструкции запуска, пользователя, эксплуатации и передачи.
- Программа ОПЭ и журнал наблюдений.
- Три приемочные проверки и протокол.
- Итоги и ретроспектива.
- Презентация PDF/PPTX и речь по участникам.

КАК ЗАГРУЗИТЬ
Заменить старые версии содержимым первой папки. Содержимое второй
папки добавить в папку команды. Архив нужен для переноса; сами
документы лучше оставить доступными для открытия на диске.
Изменения на Google Drive автоматически не вносились.

ЧТО НЕЛЬЗЯ СЧИТАТЬ ЗАКРЫТЫМ ТОЛЬКО ИЗ-ЗА НАЛИЧИЯ ФАЙЛА
Настоящая доска YouTrack и ее история, запуск другим человеком,
общая фасилитированная встреча, завершенная ОПЭ и решение по приемке.
Подготовлены формы и известные факты; чужие подписи не заполнялись.

В комплекте нет личных SQLite-баз, весов моделей, cookie и ключей.
'''
    (OUT/'00 - Что загрузить.txt').write_text(text,encoding='utf-8')
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,OUT.name+'/'+p.relative_to(OUT).as_posix())
    print(OUT)


if __name__=='__main__':build()
