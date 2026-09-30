"""Build the portfolio notebook; execute with scripts/run_notebook.py."""
import argparse
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(s):
    cell = nbf.v4.new_markdown_cell(s.strip())
    cell.metadata.update({'id': f'cell-{len(cells) + 1}', 'language': 'markdown'})
    cells.append(cell)

def code(s):
    cell = nbf.v4.new_code_cell(s.strip())
    cell.metadata.update({'id': f'cell-{len(cells) + 1}', 'language': 'python'})
    cells.append(cell)

md('''
# Энергетика в новостях: что менялось в 2022–2024 годах
**Автор: Лилия Ерофеевская**

Я разбираю, как менялись частота и тональность шести энергетических тем в GDELT. Меня интересуют два вопроса: какие темы двигались вместе и какие изменения совпадали с крупными событиями.

## Коротко о результатах

Зелёная энергетика и климатическая политика тесно связаны по частоте упоминаний, но заметно слабее — по тональности. Вблизи 24 февраля 2022 года модель выделяет рост конфликтного и санкционного счётчиков. Вокруг COP наиболее заметный положительный сдвиг приходится на первую неделю саммита. А вот предсказательная польза тональности остаётся открытым вопросом: VAR не проходит проверку остатков.

Дальше — расчёты и ограничения каждого вывода. Я работаю с готовыми агрегатами; точность исходного отбора статей здесь не подтверждена.
''')
md('''
## 1. Данные и метод

Основной файл — `data/raw/gdelt_energy_daily.csv`: один день, шесть счётчиков `count` и шесть средних значений `tone`. Отдельно сохранена географическая агрегация `country_article_counts.csv`.

- **Частота** — значение тематического счётчика в выгрузке. Без исходных записей я не могу проверить, равен ли он числу уникальных статей.
- **Тональность** — средний AvgTone документов: оценка текста целиком, а не отношения читателей или автора к отдельной технологии.
- Темы пересекаются. Их нельзя складывать как независимые части новостного потока. Общего числа новостей за день нет, поэтому доли медиаповестки я не рассчитываю.
- Названия «санкции и энергетика» и «конфликты и энергетика» условны: старые регулярные выражения не проверены по текстам статей. «Нефть и газ» не означает отдельный отбор новостей о России.
- [SQL в проекте](../sql/gdelt_energy_unified.sql) — реконструкция. Совпадение с числами CSV не подтверждено. Для запуска ноутбука BigQuery не нужен.
- Часовой пояс исходной агрегации не задокументирован: использую даты как они записаны в CSV.

Сначала проверяю таблицы и рассматриваю динамику, затем оцениваю связи и четыре гипотезы. Даты и темы выбраны после знакомства с данными. Поэтому все тесты исследовательские; причинные эффекты без контрольной группы я не оцениваю.
''')
code('''
from pathlib import Path
import os, sys, warnings
ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.matplotlib-cache'))
warnings.filterwarnings('ignore', message='adfuller currently returns a plain tuple', category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display, Markdown
from src.analysis import (TOPICS, load_daily_data, summarize_quality, benjamini_hochberg,
    differenced_topic_correlations, event_study_hac, event_study_level_hac)
from src.advanced_models import multi_event_distributed_lag, cop_event_time_model, var_changes_by_topic
LABELS = dict(zip(TOPICS, ['Нефть и газ', 'Ядерная энергетика', 'Зелёная энергетика',
    'Климатическая политика', 'Санкции и энергетика', 'Конфликты и энергетика']))
BLUE = '#356a96'
CMAP = LinearSegmentedColormap.from_list('signed', [BLUE, '#fafafa', '#bc7132'])
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10})
FIGURES, TABLES = ROOT/'reports/figures', ROOT/'reports/tables'
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
def note(s): display(Markdown(s))
def finish(fig, name, rect=None):
    if name in ('02_frequency','03_tone'):
        for ax in fig.axes:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%Y'))
            ax.set_xlim(pd.Timestamp('2022-01-01'), pd.Timestamp('2024-12-31'))
    fig.tight_layout(rect=rect)
    fig.savefig(FIGURES/(name+'.png'), dpi=130, bbox_inches='tight')
    plt.show()
    plt.close(fig)
def heatmap(matrix, title, name):
    fig, ax = plt.subplots(figsize=(9,7))
    im = ax.imshow(matrix, cmap=CMAP, vmin=-1, vmax=1)
    labels = [LABELS[c.rsplit('_',1)[0]] for c in matrix.columns]
    ax.set_xticks(range(6), labels, rotation=40, ha='right')
    ax.set_yticks(range(6), labels)
    ax.grid(False)
    for i in range(6):
        for j in range(6):
            ax.text(j,i,f'{matrix.iloc[i,j]:.2f}',ha='center',va='center',
                    color='white' if abs(matrix.iloc[i,j])>.7 else '#202020')
    ax.set_title(title+'\\nДневные значения, 2022–2024')
    fig.colorbar(im, ax=ax, shrink=.8, label='Pearson r')
    finish(fig,name)
data = load_daily_data(ROOT/'data/raw/gdelt_energy_daily.csv')
country = pd.read_csv(ROOT/'data/raw/country_article_counts.csv')
count_cols, tone_cols = [[t+'_'+s for t in TOPICS] for s in ('count','tone')]
''')
md('### 1.1. Проверка таблиц')
code('''
quality = summarize_quality(data)
assert quality.rows == 1096 and quality.missing_days == 0
assert quality.start_date == pd.Timestamp('2022-01-01') and quality.end_date == pd.Timestamp('2024-12-31')
assert quality.duplicate_dates == 0 and quality.missing_values == 0
assert country.country_code.notna().all() and country.country_code.is_unique
assert country.country_code.str.fullmatch('[A-Z]{2}').all()
assert (country.article_count >= 0).all() and (country.article_count % 1 == 0).all()
display(pd.Series(quality.__dict__, name='Результат').to_frame())
display(data.head(3))
''')
md('''
**Что я проверила.** В таблице 1 096 дней без разрывов, повторных дат и пропусков. Счётчики — неотрицательные целые числа, тональность лежит в диапазоне −100…100. Это позволяет работать с дневными рядами, но не доказывает полноту покрытия GDELT или точность тематических словарей.
''')
md('### 1.2. Географические упоминания')
code('''
top = country.nlargest(15,'article_count').sort_values('article_count')
fig,ax = plt.subplots(figsize=(9,6))
ax.barh(top.country_code,top.article_count/1e6,color=BLUE)
ax.set(title='Топ-15 кодов в географической выгрузке, 2022–2024',
       xlabel='Счётчик записей о локациях, млн',ylabel='Код исходного CSV')
finish(fig,'01_geography')
leaders = country.nlargest(3,'article_count')
note('**Что видно.** Самые большие счётчики: '+ '; '.join(
    f'{r.country_code} — {r.article_count/1e6:.2f} млн' for r in leaders.itertuples())+'. '
    'В сохранившемся SQL одна статья может учитываться несколько раз при разворачивании локаций. '
    'Регулярное выражение также ограничивает набор извлекаемых записей. Поэтому я не подменяю коды названиями стран '
    'и не называю результат числом уникальных статей или географией СМИ. В H1–H4 эта таблица не используется.')
''')
md('''
## 2. Динамика
### 2.1. Как менялась частота

На графиках использую среднее за последние 14 дней. Оно сглаживает недельный ритм и немного запаздывает. В моделях остаются исходные дневные значения.
''')
code('''
fig,axes = plt.subplots(3,2,figsize=(13,10),sharex=True)
for ax,t in zip(axes.flat,TOPICS):
    ax.plot(data.date,data[t+'_count'].rolling(14).mean(),color=BLUE)
    ax.axvline(pd.Timestamp('2022-02-24'),color='#555555',ls=':',lw=1)
    ax.set(title=LABELS[t],ylabel='Счётчик в день',ylim=(0,None))
fig.suptitle('Частота: среднее за последние 14 дней, 2022–2024\\nПунктир — 24.02.2022; шкалы панелей различаются',y=0.99)
finish(fig,'02_frequency',rect=[0,0,1,0.91])
annual = data.groupby(data.date.dt.year)[count_cols].mean()
display(annual.rename(columns={t+'_count':LABELS[t] for t in TOPICS}).round(0))
change = (annual.loc[2024]/annual.loc[2022]-1)*100
note(f'**Мой вывод.** Средний дневной санкционный счётчик в 2024 году ниже, чем в 2022-м, на '
    f'{abs(change["sanctions_energy_count"]):.1f}%, а счётчик нефти и газа — на {abs(change["fossil_energy_count"]):.1f}%. '
    'У климатической политики максимум годового среднего приходится на 2023 год. '
    'Я сравниваю средние за день, а не суммы: в 2024 году на один день больше. '
    'Без общего объёма новостей эти изменения нельзя назвать изменением доли внимания СМИ. '
    'На нескольких рядах видны резкие изменения около границ 2023 и 2024 годов. '
    'По агрегатам нельзя отделить смену новостной повестки от изменения покрытия или состава выгрузки.')
''')
md('### 2.2. Каким был эмоциональный фон')
code('''
fig,axes = plt.subplots(3,2,figsize=(13,9),sharex=True,sharey=True)
for ax,t in zip(axes.flat,TOPICS):
    ax.plot(data.date,data[t+'_tone'].rolling(14).mean(),color=BLUE)
    ax.axhline(0,color='#555555',lw=.8)
    ax.set(title=LABELS[t],ylabel='AvgTone')
fig.suptitle('Тональность: среднее дневных AvgTone за последние 14 дней, 2022–2024',y=0.99)
finish(fig,'03_tone',rect=[0,0,1,0.94])
means = data[tone_cols].mean()
note(f'**Мой вывод.** Зелёная энергетика выделяется положительным средним дневным фоном '
    f'({means["green_energy_tone"]:.2f} AvgTone), санкционная тема — наиболее отрицательным '
    f'({means["sanctions_energy_tone"]:.2f}). Каждый день здесь имеет одинаковый вес. '
    'Это не среднее по всем статьям: число документов с валидным tone неизвестно. '
    'Отрицательный тон также не означает неприятие энергетики — текст может обсуждать войну, риски и потери.')
''')
md('''
## 3. Какие показатели движутся вместе
### 3.1. Heatmap частоты

Pearson r показывает линейную связь. Значения около 1 означают совместное движение, около 0 — слабую линейную связь. Диагональ всегда равна 1.
''')
code('''
count_corr = data[count_cols].corr()
heatmap(count_corr,'Корреляции частоты упоминаний','04_count_heatmap')
''')
md('''
**Как я читаю heatmap.** Самая сильная пара — зелёная энергетика и климатическая политика (r = 0.89). Почти столь же тесно движутся нефть и газ и конфликтная тема (r = 0.87). У санкций и климатической политики связь слабая (r = −0.08).

Две сильные пары могут отражать общую повестку и пересечение документов. Общий тренд и недельный ритм тоже повышают корреляцию. Из этой картинки я не делаю вывод, что одна тема вызывает рост другой.
''')
md('### 3.2. Heatmap тональности')
code('''
tone_corr = data[tone_cols].corr()
heatmap(tone_corr,'Корреляции дневной тональности','05_tone_heatmap')
''')
md('''
**Что меняется по сравнению с частотой.** Самая заметная связь снова у нефти и газа и конфликтной темы (r = 0.66). Но у зелёной энергетики и климатической политики связь тональности всего 0.30 против 0.89 для частоты. Совместный рост числа упоминаний не обязательно сопровождается одинаковым эмоциональным фоном. У климатической и санкционной тональности линейная связь практически нулевая.
''')
md('''
### 3.3. Связаны ли изменения частоты и тона внутри одной темы

Здесь сравниваю изменения **одной и той же темы за один и тот же день**: вырос ли счётчик одновременно с тоном или снизился вместе с ним. Смотрю не на сами значения, а на их изменение со вчерашнего дня. Коэффициент Spearman ρ показывает, совпадает ли направление этих изменений: положительная связь означает, что они чаще меняются в одну сторону, отрицательная — что в разные.

**Что такое HAC.** Соседние дни могут быть похожи, например из-за недельного ритма. Поэтому нельзя считать, что каждый день даёт совершенно независимую информацию: без поправки уверенность в результате может выглядеть завышенной. HAC — способ рассчитать неопределённость так, чтобы учесть связь между близкими по времени днями и возможное изменение разброса данных. Здесь проверяется, как результат выглядит при учёте зависимости до 14 и до 28 дней.

Поправка Benjamini–Hochberg отдельно для каждого такого расчёта учитывает, что одновременно проверяются шесть тем, и снижает риск принять случайную находку за настоящую связь. Само сравнение изменений день ко дню не устраняет зависимость между соседними днями — для этого и нужна HAC-поправка.
''')
code('''
correlations = differenced_topic_correlations(data)
display(correlations.set_index('topic').rename(index=LABELS).style.format(
    {c:'{:.2e}' for c in correlations if c.startswith(('p_','q_'))},precision=3))
correlations.to_csv(TABLES/'daily_change_correlations.csv',index=False)
''')
md('''
**Мой вывод.** Наиболее заметны положительные связи у климатической политики (ρ = 0.27) и зелёной энергетики (ρ = 0.23). При росте счётчиков тон чаще сдвигается в положительную сторону, но связь умеренная. У санкционной и конфликтной тем коэффициенты по модулю меньше 0.10: небольшое p-value не делает связь сильной. Для нефти и газа и ядерной энергетики убедительной одновременной связи нет. Это ещё не проверка опережения — к лагам я вернусь в H4.
''')
md('''
## 4. Проверка гипотез
### H1. Что изменилось вблизи 24 февраля 2022 года

Модель сравнивает показатели до и после 24 февраля и оценивает, насколько они изменились относительно прежнего тренда. В ней также учтено, что после этой даты тренд мог измениться, и что показатели отличаются по дням недели. Это оценка совпавшего с датой сдвига, а не доказательство причины.

При расчёте неопределённости используется HAC: эта поправка учитывает, что соседние дни могут быть похожи, и корректирует уверенность в оценке. Здесь учитывается зависимость на срок до 14 дней.

Для счётчиков модель использует `log(1 + count)`, поэтому указанные проценты относятся к `count + 1`, а не напрямую к числу упоминаний. Изменение тональности выражается в пунктах AvgTone. 95%-й интервал показывает диапазон правдоподобных значений оценки; q-value — статистическую значимость после поправки на четыре проверки H1. Для проверки чувствительности результата сравниваются также более короткое и длинное окна: ±30 и ±90 дней.

Хотя основное окно называется ±60 дней, данных до события в нём только 54 дня: ряд начинается 1 января 2022 года. После события берутся 61 день, включая сам день 24 февраля.
''')
code('''
rows, sensitivity = [], []
for t in ('conflict_energy','sanctions_energy'):
    for suffix in ('count','tone'):
        function = event_study_hac if suffix=='count' else event_study_level_hac
        for window in (30,60,90):
            r = function(data,t+'_'+suffix,'2022-02-24',window_days=window)
            r = {k.replace('_pct',''):v for k,v in r.items()}
            r.update(topic=LABELS[t],unit='%' if suffix=='count' else 'AvgTone',window=window)
            sensitivity.append(r)
            if window==60: rows.append(r.copy())
h1 = pd.DataFrame(rows)
h1['q_value'] = benjamini_hochberg(h1.p_value_hac)
h1_sensitivity = pd.DataFrame(sensitivity)
display(h1[['topic','unit','level_change','ci_low','ci_high','q_value','n_days']].style.format({'q_value':'{:.2e}'},precision=2))
display(h1_sensitivity.pivot(index='outcome',columns='window',values='level_change').round(2))
h1.to_csv(TABLES/'h1.csv',index=False)
h1_sensitivity.to_csv(TABLES/'h1_window_sensitivity.csv',index=False)
''')
md('''
**Мой вывод.** В основной спецификации конфликтный счётчик показывает сдвиг примерно +98%, санкционный — +247% относительно продолжения локального тренда. Конфликтная тональность становится отрицательнее примерно на 0.81 пункта; отдельного сдвига санкционного тона модель не выделяет.

Размер оценки зависит от окна. Поэтому я не трактую проценты как точный «эффект войны»: события конца февраля и марта перекрываются, а предыстория короткая. Сдвиг на границе также не равен среднему изменению за весь период после события.
''')
md('''
### H2. Менялось ли внимание к ядерной и зелёной энергетике около крупных событий

В одной модели учтены все восемь событий. Для каждого события отдельно рассматриваются три периода: первая неделя (0–7 дней), вторая неделя (8–14 дней) и следующие две недели (15–30 дней).

Модель учитывает и другие закономерности: недавние значения счётчика (за день и за неделю до этого), общий тренд, различия между днями недели и сезонные изменения в течение года. Поэтому каждый коэффициент показывает, насколько счётчик в конкретном периоде отличается от ожидаемого с учётом этих факторов. Это не суммарное изменение за месяц. Проценты рассчитаны для `count + 1`, поэтому их нельзя складывать, чтобы получить общий процент за месяц.

REPowerEU датирован 18.05.2022, двенадцатый пакет санкций ЕС — 18.12.2023. [Источники дат](../reports/event_sources.md). Этот перечень — ориентиры исследования, а не полный список энергетических кризисов.
''')
code('''
ENERGY_EVENTS = {
    '2022-02-24':'Вторжение России в Украину', '2022-03-08':'Эмбарго США',
    '2022-05-18':'План REPowerEU', '2022-09-26':'Взрывы Nord Stream',
    '2022-12-05':'Введение потолка цен', '2023-04-02':'Объявление ОПЕК+',
    '2023-10-07':'Нападение ХАМАС', '2023-12-18':'12-й пакет санкций ЕС'}
coefs,joints = [],[]
for t in ('nuclear_power','green_energy'):
    c,j = multi_event_distributed_lag(data,t+'_count',ENERGY_EVENTS)
    coefs.append(c.assign(topic=t)); joints.append(j.assign(topic=t))
h2,h2_joint = pd.concat(coefs,ignore_index=True),pd.concat(joints,ignore_index=True)
# Families span both topics: 48 coefficients and 16 joint tests.
h2['p_value_fdr'] = benjamini_hochberg(h2.p_value)
h2_joint['joint_p_fdr'] = benjamini_hochberg(h2_joint.joint_p_value)
display(h2_joint.pivot(index='event',columns='topic',values='joint_p_fdr').style.format('{:.2e}'))
h2.to_csv(TABLES/'h2_lag_coefficients.csv',index=False)
h2_joint.to_csv(TABLES/'h2_joint_tests.csv',index=False)
fig,axes = plt.subplots(1,2,figsize=(14,7),sharey=True)
limit = max(10,np.ceil(h2.effect_pct.abs().max()/10)*10)
for ax,t in zip(axes,('nuclear_power','green_energy')):
    part = h2.loc[h2.topic==t]
    m = part.pivot(index='event',columns='lag_bin',values='effect_pct').reindex(list(ENERGY_EVENTS.values()))[['0–7','8–14','15–30']]
    q = part.pivot(index='event',columns='lag_bin',values='p_value_fdr').reindex(m.index)[m.columns]
    ax.imshow(m,cmap=CMAP,vmin=-limit,vmax=limit,aspect='auto')
    ax.grid(False)
    ax.set_xticks(range(3),m.columns); ax.set_yticks(range(8),m.index)
    ax.set(title=LABELS[t],xlabel='Дней после события')
    for i in range(8):
        for j in range(3):
            star = '*' if q.iloc[i,j]<.05 else ''
            ax.text(j,i,f'{m.iloc[i,j]:+.1f}{star}',ha='center',va='center',fontsize=9)
fig.suptitle('H2: условное изменение count + 1, %; 2022–2024\\n* q < 0.05 среди 48 коэффициентов; 95% интервалы сохранены в CSV',y=0.99)
finish(fig,'06_h2',rect=[0,0,1,0.90])
note(f'**Мой вывод.** Хотя бы одно из трёх окон отличается от обычного уровня для '
    f'{int((h2_joint.joint_p_fdr<.05).sum())} из 16 сочетаний «событие — тема». '
    f'В отдельных окнах после поправки на множество проверок статистически заметны '
    f'{int(((h2.effect_pct>0)&(h2.p_value_fdr<.05)).sum())} положительных и '
    f'{int(((h2.effect_pct<0)&(h2.p_value_fdr<.05)).sum())} отрицательных отклонений. '
    'Это не показывает единого устойчивого роста интереса к альтернативной энергетике после каждого кризиса. '
    'На heatmap звёздочка отмечает результат, который остаётся заметным после поправки. '
    'Крупный коэффициент без звёздочки оценён недостаточно точно, поэтому уверенно трактовать его нельзя. '
    'События февраля и марта близки по времени, и их влияние трудно разделить. '
    'Поправка уменьшает риск случайных находок, но не устраняет смещение из-за выбора событий после знакомства с данными.')
''')
md('''
**Отдельная оговорка к H2.** Крупный ядерный коэффициент через 8–14 дней после объявления ОПЕК+ совпадает по окну с закрытием последних АЭС Германии 15.04.2023. Это событие не включено в модель отдельным контролем. Поэтому приписывать весь сдвиг решению ОПЕК+ нельзя; [источник даты закрытия](../reports/event_sources.md) позволяет проверить это пересечение.

### H3. Растёт ли климатическое внимание вокруг COP

Объединяю COP27, COP28 и COP29 по времени относительно открытия. База сравнения — дни вне пяти указанных окон, с поправкой на тренд, сезонность и лаги 1 и 7. Коэффициенты общие для трёх саммитов: они не доказывают одинаковую реакцию на каждый COP. Даты открытия — из [UNFCCC](../reports/event_sources.md).
''')
code('''
COP_EVENTS = {'2022-11-06':'COP27','2023-11-30':'COP28','2024-11-11':'COP29'}
h3 = cop_event_time_model(data,'climate_policy_count',COP_EVENTS)
display(h3.style.format({'p_value':'{:.2e}','p_value_fdr':'{:.2e}'},precision=2))
h3.to_csv(TABLES/'h3.csv',index=False)
fig,ax = plt.subplots(figsize=(9,5))
ax.errorbar(['−30…−15','−14…−1','0…6','7…14','15…30'],h3.effect_pct,
    yerr=[h3.effect_pct-h3.ci_low_pct,h3.ci_high_pct-h3.effect_pct],fmt='o',color=BLUE,capsize=4)
ax.axhline(0,color='#555555',lw=1)
ax.set(title='H3: климатическая повестка вокруг COP27–COP29\\nУсловные отклонения и обычные 95% интервалы',
       xlabel='Дни относительно открытия',ylabel='Изменение count + 1, %')
finish(fig,'07_h3')
''')
md('''
**Мой вывод.** В две недели до открытия модель оценивает отклонение +6.3%, в первую неделю — +18.7%; обе оценки проходят FDR-поправку по пяти интервалам. Через 15–30 дней коэффициент становится отрицательным. Это согласуется с ростом внимания перед саммитом и в начале встречи, затем с его затуханием.

Но три COP — небольшое число событий, и все они проходят примерно в один сезон. Я считаю результат описанием этого периода, а не универсальным законом. Это условные коэффициенты при фиксированных прошлых значениях, а не общий прирост публикаций за саммит.
''')
md('''
### H4. Опережает ли тональность изменение частоты

Для всех тем использую первые разности `log(1 + count)` и AvgTone. Проверяю их стационарность по ADF, затем выбираю лаг VAR по AIC (до 14 дней). Тест Грейнджера проверяет дополнительную информацию в прошлых изменениях тона при учёте собственных лагов частоты. Это проверка внутри выборки, а не качества прогноза на новых данных.

Отдельно проверяю устойчивость VAR и остаточную автокорреляцию до 28 дней. Без этой диагностики номинальная значимость может вводить в заблуждение. Автоматический переход к VECM не используется: непринятие стационарности в уровнях само по себе не доказывает интегрированность первого порядка.
''')
code('''
h4 = var_changes_by_topic(data)
display(h4.set_index('topic').rename(index=LABELS).style.format(
    {c:'{:.2e}' for c in h4 if c.startswith('adf_p') or c.endswith(('_p','_q'))},precision=3))
h4.to_csv(TABLES/'h4_diagnostics.csv',index=False)
note(f'**Мой вывод.** После поправки на шесть проверок у {int((h4.tone_to_count_q<.05).sum())} тем результаты выглядят статистически заметными. '
    f'Но во всех {int((h4.status!="diagnostics passed").sum())} моделях осталась зависимость в остатках: модель не объяснила часть закономерностей в данных. '
    'Поэтому эти p-value нельзя считать надёжным подтверждением того, что тон помогает предсказывать частоту. '
    'Для этого нужна модель, которая лучше проходит проверки, и оценка прогноза на отдельном периоде данных, '
    'не использованном при её настройке.')
''')
md('''
Столбцы `tone_to_count_p` и `tone_to_count_q` отвечают на вопрос: помогают ли прошлые изменения тональности объяснить изменения частоты упоминаний, если модель уже учитывает прошлые значения самой частоты.

После поправки на одновременную проверку шести тем смотрим на `tone_to_count_q`:

- У пяти тем q-value меньше 0.05. По принятому порогу связь выглядит статистически заметной.
- У темы «Санкции и энергетика» q = 0.0758, то есть выше 0.05. Для неё убедительной связи таблица не показывает.

Это связь внутри имеющихся данных. Она сама по себе не означает, что тональность вызывает изменение частоты, и ещё не говорит, насколько хорошо модель предсказывает будущие значения.

### Почему H4 не считается подтверждённой

Для модели важны два диагностических столбца:

- `stable` — все шесть значений `True`: модели устойчивы по этой проверке.
- `whiteness_p` — во всех шести строках значения очень малы. Проверка показывает, что ошибки модели всё ещё связаны между соседними днями. Иначе говоря, после расчёта в остатках сохраняется временной рисунок, который модель не объяснила.

Здесь важное уточнение: в столбце `status` написано «зависимость в остатках или неустойчивость», но `stable=True` во всех строках. Значит, в этих результатах проблема именно с зависимостью ошибок во времени, а не с неустойчивостью модели.

Когда такая зависимость остаётся, оценки значимости могут быть ненадёжными: модель недостаточно хорошо описывает данные, и малое q-value не снимает эту проблему. Поэтому аккуратный вывод не «тон точно предсказывает частоту», а «внутри модели есть статистически заметный сигнал, но диагностика не позволяет уверенно на него опереться».

`adf_p_count_change` и `adf_p_tone_change` — результаты проверки преобразованных рядов. Очень маленькие значения говорят, что после перехода к изменениям от дня ко дню ряды проходят тест на стационарность. `lag` показывает, сколько прошлых дней модель включила: здесь от 13 до 14.

Для проверки прогностической пользы нужны другая спецификация и отдельный тестовый период.
''')
md('''
### Дополнительная проверка: помогает ли тон предсказывать частоту

Чтобы проверить прогноз, данные объединяются по полным неделям: недельное число упоминаний — сумма дневных счётчиков, недельный тон — среднее дневных значений. Модели выбираются только на неделях 2022–2023 годов; 2024 год остаётся отдельной проверкой.

Сравниваются две модели для прогноза изменения недельного `log(1 + count)`: базовая AR использует только прошлые изменения счётчика, а VAR дополнительно использует прошлые изменения тона. Число лагов выбирается на обучающем периоде. Для VAR среди вариантов выбирается модель с устойчивой динамикой и без выявленной остаточной автокорреляции; такие же проверки показаны для базовой модели.

На тестовом периоде строится прогноз на одну неделю вперёд по скользящей схеме: после каждой недели её фактические значения доступны для следующего прогноза, но параметры моделей не переобучаются. Ошибку считаю на уровне недельного `log(1 + count)`. Интервал возможного выигрыша VAR оцениваю блочным bootstrap с блоками по четыре недели. Это исследовательская проверка на одном отложенном годе, не гарантия будущей точности.
''')
code('''
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox

forecast_data = data.set_index('date')
forecast_rows = []
forecast_rng = np.random.default_rng(20260930)
for topic in TOPICS:
    count_col, tone_col = topic+'_count', topic+'_tone'
    weekly = forecast_data[[count_col, tone_col]].resample('W-SUN').agg(
        {count_col:'sum', tone_col:'mean'})
    full_week = forecast_data[count_col].resample('W-SUN').size().eq(7)
    weekly = weekly.loc[full_week]
    log_count = np.log1p(weekly[count_col])
    changes = pd.DataFrame({
        'count':log_count.diff(),
        'tone':weekly[tone_col].diff(),
    }).dropna()
    train = changes.loc[changes.index < '2024-01-01']
    test = changes.loc[changes.index >= '2024-01-01']

    ar_candidates = []
    for lag in range(1,9):
        fit = AutoReg(train['count'],lags=lag,trend='c').fit()
        white_p = float(acorr_ljungbox(fit.resid,lags=[12],return_df=True)['lb_pvalue'].iloc[0])
        ar_candidates.append((fit,lag,white_p))
    ar_passing = [candidate for candidate in ar_candidates if candidate[2]>=.05]
    ar_fit, ar_lag, ar_white_p = min(
        ar_passing or ar_candidates,key=lambda candidate:candidate[0].aic)

    var_candidates = []
    for lag in range(1,9):
        fit = VAR(train).fit(lag,trend='c')
        stable = bool(fit.is_stable())
        white_p = float(fit.test_whiteness(nlags=12,adjusted=True).pvalue)
        var_candidates.append((fit,lag,stable,white_p))
    var_passing = [candidate for candidate in var_candidates
                   if candidate[2] and candidate[3]>=.05]
    var_fit, var_lag, var_stable, var_white_p = min(
        var_passing or var_candidates,key=lambda candidate:candidate[0].aic)

    count_history = train['count'].to_numpy().tolist()
    pair_history = train[['count','tone']].to_numpy().tolist()
    ar_predictions, var_predictions = [],[]
    ar_params = ar_fit.params.to_numpy()
    for actual in test[['count','tone']].to_numpy():
        ar_predictions.append(ar_params[0]+sum(
            ar_params[offset+1]*count_history[-(offset+1)]
            for offset in range(ar_lag)))
        var_predictions.append(var_fit.forecast(
            np.asarray(pair_history[-var_lag:]),steps=1)[0,0])
        count_history.append(float(actual[0]))
        pair_history.append([float(actual[0]),float(actual[1])])

    actual_level = log_count.loc[test.index].to_numpy()
    previous_level = log_count.shift(1).loc[test.index].to_numpy()
    error_ar = previous_level+np.asarray(ar_predictions)-actual_level
    error_var = previous_level+np.asarray(var_predictions)-actual_level
    rmse_ar = float(np.sqrt(np.mean(error_ar**2)))
    rmse_var = float(np.sqrt(np.mean(error_var**2)))

    bootstrap_skill = []
    test_size, block_size = len(test),4
    for _ in range(2000):
        starts = forecast_rng.integers(
            0,test_size,size=int(np.ceil(test_size/block_size)))
        indices = np.concatenate([
            (start+np.arange(block_size))%test_size for start in starts
        ])[:test_size]
        sampled_ar = np.sqrt(np.mean(error_ar[indices]**2))
        sampled_var = np.sqrt(np.mean(error_var[indices]**2))
        bootstrap_skill.append(100*(1-sampled_var/sampled_ar))

    forecast_rows.append({
        'topic':topic,'train_weeks':len(train),'test_weeks':len(test),
        'ar_lag':ar_lag,'ar_whiteness_p':ar_white_p,
        'var_lag':var_lag,'var_stable':var_stable,
        'var_whiteness_p':var_white_p,
        'rmse_count_only':rmse_ar,'rmse_count_plus_tone':rmse_var,
        'rmse_skill_pct':100*(1-rmse_var/rmse_ar),
        'skill_ci_low_pct':float(np.quantile(bootstrap_skill,.025)),
        'skill_ci_high_pct':float(np.quantile(bootstrap_skill,.975)),
        'residual_checks_passed':bool(ar_white_p>=.05 and var_stable and var_white_p>=.05),
    })

h4_forecast = pd.DataFrame(forecast_rows)
display(h4_forecast.set_index('topic').rename(index=LABELS).style.format({
    'ar_whiteness_p':'{:.3f}','var_whiteness_p':'{:.3f}',
    'rmse_count_only':'{:.3f}','rmse_count_plus_tone':'{:.3f}',
    'rmse_skill_pct':'{:+.1f}%','skill_ci_low_pct':'{:+.1f}%','skill_ci_high_pct':'{:+.1f}%',
}))
h4_forecast.to_csv(TABLES/'h4_forecast_holdout.csv',index=False)
''')
md('''
**Результат.** Все шесть выбранных AR- и VAR-моделей проходят заданные проверки остатков на обучающем периоде. На 52 неделях 2024 года добавление тона уменьшило RMSE только у двух тем, а у четырёх увеличило. Блочный 95%-й интервал выигрыша не включает ноль только у климатической политики; это небольшое улучшение, и поправка за сравнение шести тем здесь не применялась. Поэтому результат не подтверждает устойчивую пользу тональности для прогноза: наблюдение охватывает один тестовый год и остаётся исследовательским.
''')
md('### 4.1. Описательная проверка разных лагов')
code('''
fig,axes = plt.subplots(3,2,figsize=(13,9),sharex=True,sharey=True)
for ax,t in zip(axes.flat,TOPICS):
    dc,dt = np.log1p(data[t+'_count']).diff(),data[t+'_tone'].diff()
    correlations_lag = [dc.corr(dt.shift(k),method='spearman') for k in range(-14,15)]
    ax.bar(range(-14,15),correlations_lag,color=BLUE)
    ax.axhline(0,color='#555555',lw=.7)
    ax.set(title=LABELS[t],ylabel='Spearman ρ',xlabel='Лаг k, дней')
fig.suptitle('Связь Δ log(1 + count) сегодня с Δ tone за t − k; 2022–2024\\nПоложительный лаг: тон измерен раньше частоты',y=0.99)
finish(fig,'08_lags',rect=[0,0,1,0.90])
''')
md('''
**Как я читаю график.** У зелёной энергетики и климатической политики заметны положительные пики около −14, −7, +7 и +14 дней. Такая повторяемость в обе стороны согласуется с общим недельным ритмом, а не только с опережением частоты тоном. Столбцы справа от нуля относятся к прошлому тону, слева — к будущему. Выбор самого высокого столбца задним числом не является проверкой гипотезы: здесь нет доверительных интервалов и поправки на просмотр всех лагов. График не отменяет проблем VAR.

## 5. Что я вынесла из анализа

**Совместная повестка не равна общему тону.** Это хорошо видно у зелёной энергетики и климатической политики: корреляция частоты около 0.89, тональности — около 0.30.

**Событийные результаты неодинаковы.** В H1 выделяются рост двух счётчиков и ухудшение конфликтного тона; изменение санкционного тона неубедительно. В H2 знак и длительность отклонения зависят от события и темы. В H3 наиболее заметно окно первой недели COP.

**Предсказательная связь пока не доказана.** В H4 номинальные p-value выглядят убедительно, но диагностика остатков меняет интерпретацию. Я оставляю эту гипотезу исследовательским вопросом.

Главная граница проекта — качество исходной выборки. На готовых CSV можно воспроизвести расчёты и описать связи. Для выводов о конкретных статьях, странах или причинном влиянии событий нужны проверенные словари, исходные документы и сопоставимый контрольный ряд. Усложнение модели этого не заменяет.

### Воспроизведение и источники

Из корня проекта: сначала `python scripts/build_notebook.py`, затем `python scripts/run_notebook.py`. Результаты сохраняются в ноутбуке, графики — в `reports/figures`, таблицы — в `reports/tables`. Исходные CSV не меняются.

- [GDELT GKG 2.1: описание полей](https://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf).
- [Реконструированный SQL](../sql/gdelt_energy_unified.sql).
- [Источники дат событий](../reports/event_sources.md).
''')
notebook = nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3'}})
nbf.validate(notebook)
out = ROOT/'notebooks/gdelt_energy_media_analysis.ipynb'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--force',
    action='store_true',
    help='overwrite the notebook even if it contains edits not present in this builder',
)
args = parser.parse_args()

if out.exists() and not args.force:
    existing = nbf.read(out, as_version=4)
    existing_content = [(cell.cell_type, cell.source) for cell in existing.cells]
    generated_content = [(cell.cell_type, cell.source) for cell in notebook.cells]
    if existing_content != generated_content:
        raise SystemExit(
            f'Refusing to overwrite {out}: its cell content differs from this builder. '
            'Sync the notebook edits into scripts/build_notebook.py first, or pass '
            '--force to intentionally replace the notebook.'
        )

nbf.write(notebook,out)
print(out)
