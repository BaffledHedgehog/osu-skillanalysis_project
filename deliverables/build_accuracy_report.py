"""Summarize the frozen accuracy experiment without publishing player histories."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from docx import Document
from docx.shared import Cm,Pt
import xlsxwriter

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'.data/accuracy-improvement'
OUT=ROOT/'deliverables'


def load(name):return json.loads((DATA/name).read_text(encoding='utf-8'))


def build():
    personal=load('personal-report.json');global_result=load('global-report.json')
    reproduced=load('benchmark-reproduction.json');refresh=load('refresh-status.json')
    evidence=personal['per_mod_ridge_comparison']['NM'];mod=personal['test']['by_mod']['NM']
    checks={}
    if (DATA/'tests.xml').exists():
        xml=ET.parse(DATA/'tests.xml')
        checks={'passed':len(xml.findall('.//testcase'))-len(xml.findall('.//skipped'))-len(xml.findall('.//failure'))-len(xml.findall('.//error')),
                'skipped':len(xml.findall('.//skipped')),'failed':len(xml.findall('.//failure'))+len(xml.findall('.//error'))}
    summary={'date':'2026-11-05','player':personal['user'],'personal_selected':personal['selected'],
             'personal_accepted':personal['accepted'],'supported_mods':personal['supported_mods'],
             'nm':evidence,'all_test':personal['test'],'cv':personal['cv'],
             'global_experiment':global_result,'refresh':refresh,'reproduction':reproduced,'checks':checks}
    (OUT/'accuracy-improvement-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    doc=Document();section=doc.sections[0]
    section.left_margin=section.right_margin=Cm(1.8);section.top_margin=section.bottom_margin=Cm(1.6)
    doc.styles['Normal'].font.name='Calibri';doc.styles['Normal'].font.size=Pt(11)
    doc.add_heading('Улучшение accuracy - pulse!',0)
    doc.add_paragraph('Университет Иннополис | Учебный проект | 05.11.2026')
    doc.add_heading('Что изменено',1)
    doc.add_paragraph('Для BaffledHedgehog подготовлена персональная модель accuracy на основе градиентного бустинга с абсолютной ошибкой. Она использует нагрузку карты, моды, прежние успешные игры и разницу между нагрузкой и показанными навыками. Вход содержит 56 признаков. Текущие pp и ник в признаки не входят.')
    doc.add_paragraph('Исправлен предварительный расчет: accuracy незавершенных провалов больше не используется как accuracy полного прохождения. При отсутствии подходящих полных результатов числовая accuracy не выдумывается. Best продолжают описывать навыки, но не считаются обычными успешными попытками при оценке доли проходов.')
    doc.add_heading('Как проверяли',1)
    doc.add_paragraph('История игрока сохранена отдельным снимком. Последняя четверть успешных обычных попыток стала проверкой. Обучение использовало только более ранние целевые результаты до 30.09.2026. Семейства проверочных карт удалены из обучающих целевых примеров. Старый Ridge и новые варианты обучались на одном раннем отрезке.')
    doc.add_paragraph('Вариант модели выбирался по трем проверкам на разных семействах карт внутри раннего отрезка. Проверялись Ridge, регрессия поправки к прогнозу по похожим картам, сам подбор похожих карт и бустинг с абсолютной ошибкой. Проверочная поздняя часть не выбирала вариант модели.')
    doc.add_paragraph('Для каждого прогноза профиль использовал только строго более ранние полные проходы. Результат целевой карты и ее семейства исключался из профиля. Во время поздней проверки ранее завершенные игры могли обновлять профиль; веса модели не менялись. Поэтому это проверка последовательного использования приложения на сохраненной истории, а не серия новых живых прогнозов.')
    doc.add_heading('Результат NM',1)
    doc.add_paragraph(f"Проверка: {evidence['samples']} игр, {evidence['families']} семейств карт. При подсчете MAE каждая песня имеет одинаковый вес, повторы одной карты не подавляют остальные.")
    table=doc.add_table(rows=1,cols=2);table.style='Table Grid'
    for c,v in zip(table.rows[0].cells,['Способ','MAE, п.п.']):c.text=v
    values=[('Среднее прежних полных проходов',mod['mean_mae_pp']),('Похожие успешные карты',mod['neighbors_mae_pp']),
            ('Прежний алгоритм Ridge',evidence['ridge_mae_pp']),('Новый градиентный бустинг',evidence['mae_pp'])]
    for name,value in values:
        cells=table.add_row().cells;cells[0].text=name;cells[1].text=f'{value:.4f}'
    improvement=(1-evidence['mae_pp']/evidence['ridge_mae_pp'])*100
    interval=evidence['difference_ci95_pp']
    doc.add_paragraph(f'Снижение MAE относительно Ridge: {improvement:.1f}%. Для NM 95% интервал разницы ошибки по повторным выборкам семейств: от {interval[0]:.3f} до {interval[1]:.3f} п.п. Отрицательные значения означают меньшую ошибку нового варианта.')
    doc.add_paragraph(f"На всех модах проверка содержит {personal['test']['samples']} игр и {personal['test']['families']} семейства. MAE нового варианта {personal['test']['mae_pp']:.4f} п.п., Ridge {personal['test']['ridge_mae_pp']:.4f} п.п. Подключен только NM: для других модов пока мало отдельных проверочных примеров.")
    doc.add_paragraph('Веса, выбранные до проверки, и сохраненные входы воспроизведены отдельно. Все прогнозы и базовый Ridge совпали с исходным прогоном; максимальное расхождение 0.0. Проверено, что даты всех игр в профиле раньше даты целевой попытки.')
    doc.add_heading('Общий вариант',1)
    doc.add_paragraph(f"Снова запрошены recent согласившихся 155 участников. Найдено {refresh['new_score_ids']} новых для локальной базы score_id; {refresh['newer_than_cutoff']} результатов датированы позже предыдущего обучения. Эти количества различаются: часть впервые полученных записей относится к более ранним играм.")
    g=global_result['fresh_test'];v=global_result['validation']
    doc.add_paragraph(f"Общий вариант {global_result['selected']} выбран по ранним обучающим игрокам. На отдельной исторической validation: {v['samples']} прохода, {v['players']} игрока; MAE {v['mae_pp']:.2f} п.п. против среднего {v['mean_mae_pp']:.2f} п.п. Это другой протокол и другая выборка, напрямую сравнивать эти числа с прежними пятью играми нельзя.")
    doc.add_paragraph(f"После фиксации выбранной модели раскрыта свежая проверка прежних тестовых игроков: {g['samples']} проходов от {g['players']} игрока. MAE {g['mae_pp']:.2f} п.п. против {g['mean_mae_pp']:.2f} п.п. у среднего. Общий вариант не подключен: данных мало и улучшение простого среднего не подтверждено.")
    doc.add_heading('Применение и ограничения',1)
    doc.add_paragraph('Персональная модель NM подключена к локальному приложению. После проверки ее веса заново обучены на доступной истории игрока. Для остальных модов и карт вне диапазона обучения остается прежний расчет. Это улучшение подтверждено для одного профиля на указанной исторической проверке; качество следующих игр и перенос на других игроков еще нужно измерить.')
    doc.add_paragraph('Веса хранятся в собственном JSON, загрузка pickle не используется. Прогноз при запросе сайта не запускает обучение. Применение к другой локальной базе разрешено только при совпадении игрока и достаточной общей истории score_id; исходная база обучения остается в метаданных.')
    if checks:doc.add_paragraph(f"Проверки кода: {checks['passed']} passed, {checks['skipped']} skipped, {checks['failed']} failed. Опциональный native host в этой работе не запускался.")
    doc.add_paragraph('Исходные данные и снимки: .data/accuracy-improvement. Они исключены из git. Команды и ограничения описаны в ACCURACY.md. Источники методов: https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html и https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.GroupKFold.html')
    doc.save(OUT/'Улучшение accuracy - pulse!.docx')
    book=xlsxwriter.Workbook(OUT/'Улучшение accuracy - pulse!.xlsx',{'strings_to_formulas':False})
    sheet=book.add_worksheet('NM');sheet.write_row(0,0,['Способ','MAE, п.п.'])
    for i,(name,value) in enumerate(values,1):sheet.write_row(i,0,[name,value])
    sheet.set_column('A:A',44);sheet.set_column('B:B',18)
    sheet=book.add_worksheet('Выбор варианта');sheet.write_row(0,0,['Вариант','MAE CV, п.п.','Игры','Семейства'])
    for i,(name,m) in enumerate(personal['cv'].items(),1):sheet.write_row(i,0,[name,m['mae_pp'],m['samples'],m['families']])
    sheet.set_column('A:A',32);sheet.set_column('B:D',20)
    book.close()
    print(f'NM MAE: {evidence["ridge_mae_pp"]:.4f} -> {evidence["mae_pp"]:.4f}; {improvement:.1f}% improvement')


if __name__=='__main__':build()
