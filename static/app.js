const state = {
  user: new URLSearchParams(location.search).get('user') || localStorage.getItem('pulse-user') || '',
  mapId: new URLSearchParams(location.search).get('map') || localStorage.getItem('pulse-map') || '',
  mods: new Set(), goal: 'comfort', maps: [], analysis: null,
};

const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[char]));
const formatTime = (seconds) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
let toastTimer;

function toast(message, error = false) {
  const element = $('#toast');
  // A modal dialog is in the browser top layer; z-index on body cannot cover it.
  const host = document.querySelector('dialog[open]') || document.body;
  if (element.parentElement !== host) host.append(element);
  element.textContent = message;
  element.classList.toggle('error', error);
  element.classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.remove('visible'), 4500);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let body;
  try { body = await response.json(); } catch { body = {}; }
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `Ошибка ${response.status}`);
  return body;
}

function currentMods() {
  return state.mods.size ? ['HD', 'HR', 'DT', 'EZ', 'HT', 'CL'].filter(mod => state.mods.has(mod)).join('') : 'NM';
}

function updateMods() {
  const label = currentMods();
  document.querySelectorAll('#mods-picker button').forEach(button => {
    button.classList.toggle('selected', button.dataset.mod === 'NM' ? label === 'NM' : state.mods.has(button.dataset.mod));
  });
}

async function loadCatalog() {
  state.maps = await api('/api/maps');
  $('#catalog-count').textContent = state.maps.length;
  const list = $('#catalog-list');
  if (!state.maps.length) {
    list.innerHTML = '<div class="empty-state panel">Каталог пока пуст. Вставь ссылку на карту или загрузи файл .osu.</div>';
    return;
  }
  list.innerHTML = state.maps.map(map => `<div class="catalog-item"><div><div class="catalog-name">${escapeHtml(map.artist)} - ${escapeHtml(map.title)}</div><div class="catalog-sub">${escapeHtml(map.version)}, ${escapeHtml(map.creator)}, ${formatTime(map.duration)}</div></div><button type="button" data-id="${escapeHtml(map.id)}">Разобрать</button></div>`).join('');
  list.querySelectorAll('button').forEach(button => button.addEventListener('click', () => selectMap(button.dataset.id)));
}

async function loadSummary() {
  const box = $('#player-summary');
  if (!state.user) {
    box.innerHTML = '<span class="mini-dot"></span> Вставь ссылку на профиль osu! или укажи ник.';
    $('#avatar').textContent = '?';
    return;
  }
  $('#avatar').textContent = state.user[0].toUpperCase();
  const summary = await api(`/api/users/${encodeURIComponent(state.user)}/summary`);
  if (!summary.plays) {
    box.innerHTML = `<span class="mini-dot"></span><strong>${escapeHtml(state.user)}</strong>. Пока нет результатов.`;
  } else {
    const recentCount = (summary.sources?.osu_api_recent || 0) + (summary.sources?.hinai_recent || 0) + (summary.sources?.osu_web_recent || 0);
    const bestCount = (summary.sources?.osu_api_best || 0) + (summary.sources?.hinai_best || 0) + (summary.sources?.osu_web_best || 0);
    box.innerHTML = `<span class="mini-dot"></span><strong>${escapeHtml(state.user)}</strong>: ${summary.plays} результатов${recentCount || bestCount ? `, ${recentCount} recent и ${bestCount} best` : ''}. Средняя точность: ${summary.accuracy}%${summary.best_pp !== null ? `. Лучший результат: ${summary.best_pp} PP` : ''}${summary.comfort_stars ? `. Ориентир: ${summary.comfort_stars} звезд` : ''}.`;
    box.innerHTML += ` Проходов: ${summary.passed}; провалов: ${summary.failed || 0}.`;
    if (summary.no_fail_plays) box.innerHTML += ` No Fail: ${summary.no_fail_plays}, они не определяют pass без NF.`;
    const unresolved = Math.max(0, (summary.unknown_outcomes || 0) - (summary.no_fail_plays || 0));
    if (unresolved) box.innerHTML += ` Результатов без подтвержденного исхода: ${unresolved}.`;
    if (summary.progress?.status === 'available') {
      const progress = summary.progress;
      if (progress.accuracy_change_pp !== null) box.innerHTML += ` На повторно сыгранных картах изменение accuracy за последние 30 дней: ${progress.accuracy_change_pp > 0 ? '+' : ''}${progress.accuracy_change_pp} п.п. (${progress.accuracy_maps} карт).`;
      if (progress.pass_rate_change_pp !== null) box.innerHTML += ` Изменение доли проходов: ${progress.pass_rate_change_pp > 0 ? '+' : ''}${progress.pass_rate_change_pp} п.п. (${progress.pass_maps} карт).`;
    }
    box.innerHTML += ' Свежие игры имеют больший вес; за 60 дней вес уменьшается вдвое.';
  }
}

function renderPrediction(prediction) {
  const box = $('#prediction-content');
  if (!prediction || prediction.pass_probability === null) {
    if (prediction?.prediction_status === 'unknown_skill_range') {
      box.innerHTML = `<div class="prediction-empty">Шанс прохождения пока неизвестен. Для такой скорости, AR или модов нет подходящих примеров в истории.<br>Модель не считает отсутствие опыта доказательством невозможности прохождения.</div><div style="margin-top:13px">${prediction.factors.map(factor => `<div class="prediction-factor">${escapeHtml(factor)}</div>`).join('')}</div>`;
      return;
    }
    const count = prediction?.observations || 0;
    box.innerHTML = `<div class="prediction-empty">${state.user ? `Есть ${count} из 3 необходимых результатов. После трёх игр появится первая персональная оценка.` : 'Укажи ник игрока и добавь минимум три результата. Тогда появится персональная оценка.'}<br><br>Сейчас можно изучать объективную нагрузку карты по участкам.</div>`;
    return;
  }
  const ml = prediction.ml_status;
  const accuracyML = prediction.accuracy_ml_status;
  const cnn = prediction.cnn_status;
  const cnnAccuracy = prediction.accuracy_model === 'validated_pattern_cnn';
  const accuracyActive = prediction.accuracy_model === 'validated_personal_ridge';
  const mlNote = prediction.model === 'personal_history_posterior'
    ? 'Прогноз уточнен по сохраненным попыткам этой версии карты с теми же модами. Качество такого уточнения еще нужно проверить на следующих играх.'
    : ml?.status === 'trained'
    ? `ML прохождения обучена на ${ml.plays} попытках и проверена на других картах (ошибка Brier: ${ml.metrics.brier}, базовый прогноз: ${ml.metrics.baseline_brier}).`
    : ml?.status === 'insufficient_mod_data'
      ? 'Для выбранных модов в обучающей истории недостаточно данных. Используется предварительный расчет.'
    : ml?.status === 'rejected'
      ? `ML пока не применяется: на отложенных картах её ошибка выше базового прогноза.`
      : `Для ML прохождения нужны 40 попыток на 10 картах, минимум 8 успехов и 8 неудач. Сейчас: ${ml?.plays || 0} попыток. Best в обучение прохождения не входят.`;
  const accuracyLabel = cnnAccuracy ? 'CNN: ТОЧНОСТЬ ПО ПАТТЕРНАМ'
    : accuracyActive
    ? (accuracyML.source_mode === 'best_scores' ? 'ML: ТОЧНОСТЬ ПО BEST-РЕЗУЛЬТАТАМ' : accuracyML.source_mode === 'local_scores' ? 'ML: ТОЧНОСТЬ ПО ЛОКАЛЬНЫМ РЕЗУЛЬТАТАМ' : 'ML: ТОЧНОСТЬ ПРИ ПРОХОЖДЕНИИ')
    : 'ТОЧНОСТЬ: ПРЕДВАРИТЕЛЬНЫЙ РАСЧЕТ';
  let accuracyNote = 'Для ML точности нужны 40 пройденных результатов на 10 картах.';
  if (cnnAccuracy) {
    const release = cnn.release.accuracy;
    accuracyNote = `CNN проверена на других семействах песен: ${cnn.accuracy_metrics.accuracy_attempts} проходов. Средняя ошибка ${release.test_metric.toFixed(2)} п.п.; у контрольной модели ${release.reference_metric.toFixed(2)} п.п. Это проверка на сохраненной истории, качество будущих игр еще нужно измерить. Диапазон рассчитан по ошибкам проверки.`;
  } else if (accuracyActive) {
    accuracyNote = `ML точности обучена на ${accuracyML.plays} результатах. На ${accuracyML.metrics.holdout_maps} картах вне обучения средняя ошибка ${accuracyML.metrics.mae_pp.toFixed(2)} п.п.; у простого среднего ${accuracyML.metrics.baseline_mae_pp.toFixed(2)} п.п.`;
    if (accuracyML.source_mode === 'best_scores') accuracyNote += ' Используются только best: оценка смещена к удачным играм и не описывает обычную попытку.';
    if (accuracyML.source_mode === 'local_scores') accuracyNote += ' Используются полные локальные результаты. Их pass неизвестен; для ML прохождения они не используются.';
    accuracyNote += ' Диапазон показан по ошибкам проверки и не является гарантией.';
  } else if (accuracyML?.status === 'out_of_range') {
    accuracyNote = 'Карта вне диапазона сложности обучающих данных. ML точности для нее не применяется.';
  } else if (accuracyML?.status === 'insufficient_mod_data') {
    accuracyNote = 'В обучающей истории нет подходящих данных по выбранным модам. Используется предварительный расчет.';
  } else if (accuracyML?.status === 'rejected') {
    accuracyNote = 'ML точности не улучшила среднее значение на отложенных картах. Используется предварительный расчет.';
  }
  box.innerHTML = `<div class="prediction-row"><div><div class="prediction-label">${prediction.model === 'personal_history_posterior' ? 'ПРОГНОЗ С УЧЕТОМ ИСТОРИИ КАРТЫ' : ml?.status === 'trained' ? 'ML: ШАНС PASS' : 'ПРЕДВАРИТЕЛЬНЫЙ ШАНС PASS'}</div><div class="prediction-number">${prediction.pass_probability}%</div></div><span class="confidence">${prediction.confidence === 'medium' ? 'Средняя уверенность' : 'Мало данных'}</span></div>
  <div class="prediction-label">${accuracyLabel}</div><div class="accuracy-num">${prediction.expected_accuracy}% <span style="font-size:11px;color:#8995ab;font-weight:400">примерно ${prediction.accuracy_range[0]}-${prediction.accuracy_range[1]}%</span></div>
  <p class="prediction-message">По ${prediction.observations} результатам. ${escapeHtml(prediction.message)}<br>${escapeHtml(mlNote)}<br><br>${escapeHtml(accuracyNote)}</p>
  <div style="margin-top:13px">${prediction.factors.map(factor => `<div class="prediction-factor">${escapeHtml(factor)}</div>`).join('')}</div>`;
}

function renderTimeline(analysis) {
  const windows = analysis.windows || [];
  const max = Math.max(1, ...windows.map(window => window.load));
  const bars = windows.map(window => {
    const total = Math.max(2, window.load / max * 100);
    const aim = window.load ? total * window.aim / window.load : 0;
    const speed = Math.max(0, total - aim);
    return `<div class="timeline-bar" title="${formatTime(window.start)}, Aim ${window.aim}, Speed ${window.speed}"><div class="aim-part" style="height:${aim}%"></div><div class="speed-part" style="height:${speed}%"></div></div>`;
  }).join('');
  $('#timeline').innerHTML = bars;
  const end = windows.length ? windows[windows.length - 1].end : analysis.duration;
  $('#timeline').insertAdjacentHTML('afterend', `<div class="timeline-axis" id="timeline-axis"><span>0:00</span><span>${formatTime(end / 2)}</span><span>${formatTime(end)}</span></div>`);
  document.querySelectorAll('#timeline-axis').forEach((axis, index) => { if (index > 0) axis.remove(); });
  $('#peak-list').innerHTML = analysis.peaks.map((peak, index) => `<div class="peak-item"><span class="peak-time">${formatTime(peak.start)}-${formatTime(peak.end)}</span><span>${index === 0 ? 'Максимальная нагрузка' : 'Сложный участок'}</span><span class="peak-type">${peak.aim >= peak.speed ? 'Aim' : 'Speed'}</span></div>`).join('');
}

function renderAnalysis(analysis) {
  state.analysis = analysis;
  $('#result-section').classList.remove('hidden');
  $('#map-difficulty').textContent = analysis.stars.toFixed(1);
  $('#map-title').textContent = `${analysis.artist} - ${analysis.title}`;
  $('#map-subtitle').textContent = `[${analysis.version}], ${analysis.creator}`;
  $('#stat-stars').textContent = `${analysis.stars.toFixed(2)}*`;
  $('#stat-aim').textContent = analysis.aim.toFixed(2);
  $('#stat-speed').textContent = analysis.speed.toFixed(2);
  $('#stat-bpm').textContent = analysis.bpm;
  $('#stat-duration').textContent = `${formatTime(analysis.duration)}, ${analysis.object_count} объектов`;
  const official = $('#official-link');
  official.classList.toggle('hidden', !/^\d+$/.test(analysis.id));
  official.href = `https://osu.ppy.sh/beatmaps/${analysis.id}`;
  $('#dialog-map-name').textContent = `${analysis.artist} - ${analysis.title} [${analysis.version}] +${analysis.mods}`;
  renderPrediction(analysis.prediction);
  renderTimeline(analysis);
  renderLocalPrediction(analysis.local_prediction);
  updateMods();
}

async function selectMap(id, scroll = true) {
  state.mapId = id;
  localStorage.setItem('pulse-map', id);
  const query = new URLSearchParams({ user: state.user, mods: currentMods() });
  const analysis = await api(`/api/maps/${encodeURIComponent(id)}/analysis?${query}`);
  renderAnalysis(analysis);
  if (analysis.prediction?.suggest_training || analysis.prediction?.too_hard) {
    $('#prediction-content').insertAdjacentHTML('beforeend', '<div id="training-maps"><h3>Похожие карты для тренировки</h3><p>Ищем похожие карты с прогнозом от 95% accuracy...</p></div>');
    try {
      const training = await api(`/api/maps/${encodeURIComponent(id)}/training?${query}`);
      const box = $('#training-maps');
      if (state.mapId === id && currentMods() === analysis.mods && box) {
        box.innerHTML = `<h3>Похожие карты для тренировки</h3><p>${escapeHtml(training.message)}</p>` + training.maps.map(map =>
          `<button class="import-button" type="button" data-training-map="${escapeHtml(map.id)}">${escapeHtml(map.artist)} - ${escapeHtml(map.title)} [${escapeHtml(map.version)}], ${map.stars}*, около ${map.prediction.expected_accuracy}% accuracy</button>`).join('');
        box.querySelectorAll('[data-training-map]').forEach(button => button.addEventListener('click', () => action(() => selectMap(button.dataset.trainingMap))));
      }
    } catch (error) { if ($('#training-maps')) $('#training-maps').textContent = error.message; }
  }
  if (scroll) $('#result-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function loadRecommendations() {
  const query = new URLSearchParams({ user: state.user, goal: state.goal, mods: currentMods() });
  const data = await api(`/api/recommendations?${query}`);
  const list = $('#recommendation-list');
  if (!data.maps.length) {
    list.innerHTML = `<div class="empty-state panel">${data.catalog_size ? 'Все добавленные карты уже есть в истории. Добавь ещё карты для подбора.' : 'Добавь карты в каталог, чтобы получить подбор.'}</div>`;
    return;
  }
  list.innerHTML = data.maps.map(map => `<article class="rec-card panel" data-id="${escapeHtml(map.id)}" tabindex="0" role="button" aria-label="Разобрать карту ${escapeHtml(map.title)}"><div class="rec-top"><span>osu!standard, +${escapeHtml(map.mods)}</span><span class="rec-stars">${map.stars.toFixed(2)}*</span></div><div class="rec-title">${escapeHtml(map.title)}</div><div class="rec-sub">${escapeHtml(map.artist)} [${escapeHtml(map.version)}]</div><div class="rec-foot"><span>${formatTime(map.duration)}, ${map.bpm} BPM</span><span class="rec-chance">${map.prediction.pass_probability === null ? 'По сложности' : `${map.prediction.pass_probability}% pass`}</span></div></article>`).join('');
  list.querySelectorAll('.rec-card').forEach(card => {
    card.addEventListener('click', () => selectMap(card.dataset.id));
    card.addEventListener('keydown', event => { if (event.key === 'Enter') selectMap(card.dataset.id); });
  });
}

async function refresh() {
  await Promise.all([loadCatalog(), loadSummary()]);
  if (state.mapId) {
    try { await selectMap(state.mapId, false); } catch { state.mapId = ''; localStorage.removeItem('pulse-map'); }
  }
  await loadRecommendations();
}

async function action(fn, busyElement) {
  const dialog = busyElement?.closest('dialog') || document.querySelector('dialog[open]');
  const feedback = dialog?.querySelector('.dialog-error');
  if (feedback) { feedback.textContent = ''; feedback.hidden = true; }
  try {
    if (busyElement) { busyElement.disabled = true; busyElement.dataset.original = busyElement.textContent; busyElement.textContent = 'Подождите...'; }
    await fn();
  } catch (error) {
    const message = error.message || 'Произошла ошибка.';
    if (feedback) { feedback.textContent = message; feedback.hidden = false; }
    toast(message, true);
  }
  finally { if (busyElement) { busyElement.disabled = busyElement.dataset.stayDisabled === 'true'; busyElement.textContent = busyElement.dataset.original; } }
}

$('#map-form').addEventListener('submit', event => {
  event.preventDefault();
  action(async () => {
    const result = await api('/api/maps/link', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ value: $('#map-input').value.trim() }) });
    await loadCatalog(); await selectMap(result.id); await loadRecommendations();
    toast('Карта добавлена и проанализирована.');
  }, $('#map-submit'));
});

$('#sample-button').addEventListener('click', () => {
  action(async () => {
    const result = await api('/api/maps/starter', { method: 'POST' });
    await loadCatalog(); await selectMap(result.maps.find(map => map.id === '221777')?.id || result.maps[0].id);
    await loadRecommendations();
    toast(`В каталоге ${result.maps.length} примера карт.`);
  }, $('#sample-button'));
});

$('#map-file').addEventListener('change', event => {
  const file = event.target.files[0]; if (!file) return;
  action(async () => {
    const form = new FormData(); form.append('file', file);
    const result = await api('/api/maps/upload', { method: 'POST', body: form });
    await loadCatalog(); await selectMap(result.maps[0].id); await loadRecommendations();
    toast(`Добавлено карт: ${result.maps.length}.`);
  });
  event.target.value = '';
});

$('#player-form').addEventListener('submit', event => {
  event.preventDefault();
  action(async () => {
    const reference = $('#player-input').value.trim();
    if (!reference) throw new Error('Укажи ссылку на профиль или ник игрока.');
    const result = await importProfile(reference);
    toast(result.cached ? 'Профиль загружен из кэша.' : `Добавлено новых попыток: ${result.imported}. Уникальных карт: ${result.unique_maps}.`);
  }, $('#player-form button[type="submit"]'));
});

async function importProfile(reference, options = {}) {
  const hint = $('#api-hint');
  const previous = hint.textContent;
  hint.textContent = 'Загружаем результаты и карты. Это может занять несколько минут...';
  try {
    const previousUser = state.user;
    const result = await api('/api/users/import', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username: reference, ...options }) });
    state.user = result.username;
    $('#player-input').value = state.user;
    localStorage.setItem('pulse-user', state.user);
    await refresh();
    if (result.highlight_map_id && (!state.mapId || previousUser !== result.username)) await selectMap(result.highlight_map_id);
    return result;
  } finally { hint.textContent = previous; }
}

$('#import-button').addEventListener('click', () => action(async () => {
  const reference = $('#player-input').value.trim() || state.user;
  if (!reference) throw new Error('Сначала укажи ссылку на профиль или ник игрока.');
  const result = await importProfile(reference, { force: true });
  toast(`Добавлено новых попыток: ${result.imported}; повторов карт: ${result.duplicates}. История дополнена.`);
}, $('#import-button')));

async function refreshSession() {
  const [session, oauth] = await Promise.all([api('/api/session/cookies'), api('/api/session/oauth')]);
  state.cookiesActive = session.active;
  state.apiConfigured = oauth.active || oauth.environment_configured;
  $('#session-status').textContent = state.apiConfigured ? 'Официальный OAuth API подключен. Он используется вместо cookie.' : session.active ? 'Формат cookie принят; сессия проверится при загрузке истории. Cookie только в памяти.' : 'Cookie и OAuth не подключены. Доступен открытый источник и локальный импорт.';
  $('#api-hint').textContent = state.apiConfigured ? 'Официальный osu! API подключен. Доступна загрузка следующих страниц результатов.' : 'Без входа: открытый API Hinamizawa. Данные профиля передаются этому сервису.';
}

function openAccessDialog(selector) {
  const dialog = $(selector);
  const error = dialog.querySelector('.dialog-error');
  if (error) { error.hidden = true; error.textContent = ''; }
  dialog.showModal();
}
$('#cookies-button').addEventListener('click', () => openAccessDialog('#cookies-dialog'));
$('#cookies-dialog').addEventListener('close', () => { $('#cookies-text').value = ''; $('#cookies-file').value = ''; });
$('#cookies-close').addEventListener('click', () => { $('#cookies-text').value = ''; $('#cookies-dialog').close(); });
$('#cookies-file').addEventListener('change', event => action(async () => {
  const file = event.target.files[0];
  if (!file) return;
  if (file.size > 256000) throw new Error('Файл cookie превышает 256 КБ.');
  $('#cookies-text').value = await file.text();
  event.target.value = '';
}));
$('#cookies-form').addEventListener('submit', event => {
  event.preventDefault();
  action(async () => {
    await api('/api/session/cookies', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ content: $('#cookies-text').value }) });
    $('#cookies-text').value = '';
    $('#cookies-dialog').close();
    await refreshSession();
    toast('Cookie импортированы. Вход проверится при загрузке истории.');
  }, $('#cookies-form button[type="submit"]'));
});
$('#cookies-clear').addEventListener('click', () => action(async () => {
  await api('/api/session/cookies', { method: 'DELETE' });
  $('#cookies-text').value = '';
  $('#cookies-dialog').close();
  await refreshSession();
  toast('Сессия удалена.');
}));
$('#oauth-button').addEventListener('click', () => openAccessDialog('#oauth-dialog'));
$('#cookies-to-oauth').addEventListener('click', () => {
  $('#cookies-dialog').close();
  openAccessDialog('#oauth-dialog');
});
$('#oauth-close').addEventListener('click', () => $('#oauth-dialog').close());
$('#oauth-dialog').addEventListener('close', () => { $('#oauth-secret').value = ''; });
$('#oauth-form').addEventListener('submit', event => {
  event.preventDefault();
  action(async () => {
    await api('/api/session/oauth', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({client_id: $('#oauth-id').value.trim(), client_secret: $('#oauth-secret').value.trim()})});
    $('#oauth-dialog').close();
    await refreshSession();
    toast('OAuth проверен. Официальный API подключен; ключи только в памяти.');
  }, $('#oauth-submit'));
});
$('#oauth-clear').addEventListener('click', () => action(async () => {
  await api('/api/session/oauth', {method: 'DELETE'});
  $('#oauth-dialog').close();
  await refreshSession();
  toast(state.apiConfigured ? 'Сессия OAuth удалена. Ключи из .env остаются настроены.' : 'OAuth отключен.');
}, $('#oauth-clear')));

$('#nf-health-close').addEventListener('click', () => $('#nf-health-dialog').close());
async function loadNfReplays(selected = null) {
  const data = await api(`/api/local/nf-replays?${new URLSearchParams({user: state.user})}`);
  const select = $('#nf-health-select');
  select.innerHTML = data.replays.map(r => `<option value="${escapeHtml(r.id)}">${escapeHtml(r.metadata.artist)} - ${escapeHtml(r.metadata.title)} [${escapeHtml(r.metadata.version)}] +${escapeHtml(r.mods)}, ${(r.accuracy * 100).toFixed(1)}%, ${escapeHtml(r.played_at.slice(0, 10))}</option>`).join('');
  if (selected) select.value = selected;
  $('#nf-health-submit').disabled = !data.replays.length;
  if (!data.replays.length) $('#nf-health-result').textContent = 'Нет сохраненных NF-реплеев. Повтори локальный импорт или загрузи .osr ниже.';
}
$('#nf-health-button').addEventListener('click', () => action(async () => {
  if (!state.user) throw new Error('Сначала выбери профиль игрока.');
  openAccessDialog('#nf-health-dialog');
  $('#nf-health-result').textContent = 'Загружаем список NF-попыток...';
  await loadNfReplays();
  if ($('#nf-health-select').options.length) $('#nf-health-result').textContent = 'Выбери попытку и запусти расчет.';
}, $('#nf-health-button')));

function renderHealth(result) {
  const valid = !!result.verified_counterfactual;
  const failedAt = result.without_nf_fail_time_ms;
  const outcome = !valid ? 'Исход без NF не подтвержден.' : failedAt === null ? 'Условие провала без NF не сработало.' : `Без NF условие провала сработало бы на ${(failedAt / 1000).toFixed(1)} секунде карты.`;
  const points = (result.hp || []).filter(p => Number.isFinite(p.time_ms) && Number.isFinite(p.hp));
  const end = Math.max(1, ...points.map(p => p.time_ms));
  const stride = Math.max(1, Math.ceil(points.length / 900));
  const path = points.filter((_, i) => i % stride === 0 || i === points.length - 1).map(p => `${(40 + p.time_ms / end * 680).toFixed(2)},${(190 - Math.max(0, Math.min(1, p.hp)) * 160).toFixed(2)}`).join(' ');
  const failLine = failedAt !== null && failedAt <= end ? `<line x1="${40 + failedAt / end * 680}" x2="${40 + failedAt / end * 680}" y1="25" y2="195" stroke="#ff748e" stroke-dasharray="5 4"/>` : '';
  $('#nf-health-result').innerHTML = `<h3>${escapeHtml(outcome)}</h3><p>${escapeHtml(result.message)}</p><p>Минимум HP: ${(result.minimum_hp * 100).toFixed(1)}%. Движок: ${escapeHtml(result.engine_version)}, ${escapeHtml(result.health_processor)}. ${result.cached ? 'Результат из кеша.' : ''}</p><svg class="hp-chart" viewBox="0 0 750 230" role="img" aria-label="График HP по времени карты"><line x1="40" x2="720" y1="190" y2="190" stroke="#6a7892"/><line x1="40" x2="720" y1="30" y2="30" stroke="#35415a"/><text x="3" y="35" fill="#b7c2d6" font-size="12">100%</text><text x="10" y="194" fill="#b7c2d6" font-size="12">0%</text><polyline points="${path}" fill="none" stroke="#70e5cb" stroke-width="2"/>${failLine}<text x="40" y="215" fill="#b7c2d6" font-size="12">0:00</text><text x="670" y="215" fill="#b7c2d6" font-size="12">${formatTime(end / 1000)}</text></svg><p>Время указано по шкале карты. При DT/HT реальная длительность отличается. Совпадение статистики проверяется по воспроизведенным попаданиям и тикам; версия движка может отличаться от версии исходной игры.</p>`;
}

$('#nf-health-form').addEventListener('submit', event => {
  event.preventDefault();
  action(async () => {
    $('#nf-health-result').textContent = 'Воспроизводим реплей без окна и считаем HP...';
    const result = await api('/api/local/nf-health', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({user: state.user, play_id: $('#nf-health-select').value})});
    renderHealth(result);
  }, $('#nf-health-submit'));
});
$('#nf-health-file').addEventListener('change', event => action(async () => {
  const file = event.target.files[0];
  if (!file) return;
  const body = new FormData();
  body.append('user', state.user); body.append('file', file);
  const registered = await api('/api/local/nf-replay', {method: 'POST', body});
  await loadNfReplays(registered.play_id);
  $('#nf-health-result').textContent = 'Реплей добавлен для анализа. Нажми "Рассчитать HP".';
  event.target.value = '';
}));
$('#more-scores-button').addEventListener('click', () => action(async () => {
  if (!state.cookiesActive && !state.apiConfigured) throw new Error('Для страниц истории подключите cookie osu! или официальный API. Открытый источник отдает только первую страницу.');
  const reference = $('#player-input').value.trim() || state.user;
  if (!reference) throw new Error('Укажите профиль игрока.');
  const result = await importProfile(reference, { force: true, max_scores: 1000 });
  toast(`Новых попыток: ${result.imported}. Сервер отдал ${result.received_recent} recent и ${result.received_best} best; полный архив сайт не гарантирует.`);
}, $('#more-scores-button')));
$('#find-client-button').addEventListener('click', () => action(async () => {
  const result = await api('/api/local/clients');
  $('#local-clients').textContent = result.clients.length ? result.clients.map(c => `${c.client}: ${c.path}${c.username ? ` (${c.username})` : ''}`).join('\n') : 'Клиент не найден. Откройте osu! или укажите папку вручную.';
  const first = result.clients.find(c => c.supported);
  if (first) $('#local-path').value = first.path;
}, $('#find-client-button')));
$('#local-import-button').addEventListener('click', () => action(async () => {
  $('#local-clients').textContent = 'Ищем клиент и новые результаты. Первое сопоставление карт может занять время...';
  const result = await api('/api/local/import', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({
    path: $('#local-path').value.trim() || null, user: $('#local-player').value.trim() || null, force: $('#local-force').checked,
  }) });
  state.user = result.username;
  localStorage.setItem('pulse-user', state.user);
  $('#player-input').value = state.user;
  await refresh();
  if (result.highlight_map_id) await selectMap(result.highlight_map_id);
  $('#local-clients').textContent = `Игрок: ${result.username}. Новых: ${result.imported}; карт не найдено: ${result.missing_maps}; файлов реплеев: ${result.available_replays}. ${result.message}`;
  toast(result.cached ? 'Локальные файлы не изменились. Использован кэш.' : `Добавлено ${result.imported} локальных результатов.`);
}, $('#local-import-button')));

$('#train-model-button').addEventListener('click', () => action(async () => {
  if (!state.user) throw new Error('Сначала загрузи профиль игрока.');
  const result = await api(`/api/users/${encodeURIComponent(state.user)}/model/train`, { method: 'POST' });
  await refresh();
  const accuracy = result.accuracy.status === 'trained' ? 'ML точности обучена' : 'ML точности пока не применяется';
  const pass = result.pass.status === 'trained' ? 'ML прохождения обучена' : 'для ML прохождения не хватает данных или качества';
  toast(`${accuracy}; ${pass}.`);
}, $('#train-model-button')));

let cnnPoll;
const cnnVariants = {context_only: 'Без последовательности нот', original: 'Оригинал', mirror_x: 'Отражение X', mirror_y: 'Отражение Y', mirror_xy: 'Отражение XY', four_views: 'Все 4 варианта'};
const cnnPhases = {starting: 'Запуск', snapshot: 'Фиксируем данные', sequences: 'Готовим последовательности', pretraining: 'Предобучение геометрии', completed: 'Готово', ...cnnVariants};

async function loadCnnStatus() {
  clearTimeout(cnnPoll);
  const user = state.user;
  const [result, local] = await Promise.all([
    api(`/api/users/${encodeURIComponent(user)}/cnn`),
    api(`/api/users/${encodeURIComponent(user)}/cnn/local`),
  ]);
  if (user !== state.user || !$('#cnn-dialog').open) return;
  $('#cnn-train').disabled = result.running || local.running || !result.available;
  $('#local-cnn-train').disabled = result.running || local.running || !local.available;
  $('#cnn-stop').disabled = !result.running && !local.running;
  for (const selector of ['#cnn-train','#local-cnn-train','#cnn-stop']) {
    $(selector).dataset.stayDisabled = String($(selector).disabled);
  }
  let content = result.available ? '' : '<p>Установите PyTorch: python -m pip install -r requirements-cnn.txt</p>';
  const progress = result.progress || {};
  if (result.running) {
    content += `<p>${escapeHtml(cnnPhases[progress.phase] || progress.phase)}: ${progress.step || 0}/${progress.total || 1}. Прошло ${Math.round(progress.elapsed_seconds || 0)} с.</p><progress value="${Number(progress.step) || 0}" max="${Number(progress.total) || 1}"></progress>`;
  } else if (progress.status === 'failed' || progress.status === 'interrupted') {
    content += `<p>${escapeHtml(progress.error || progress.message || 'Обучение прервалось. Можно повторить запуск.')}</p>`;
  }
  const report = result.report;
  $('#cnn-report').hidden = !report;
  $('#cnn-report').href = `/api/users/${encodeURIComponent(user)}/cnn/report`;
  if (result.forward) {
    content += `<p>${result.forward.status === 'awaiting_new_plays' ? 'Новых попыток после сохранения модели пока нет.' : `Проверка после сохранения модели: ${result.forward.attempts} попыток. Результаты доступны в отчете.`}</p>`;
  }
  if (report) {
    content += `<p>Реальных попыток: ${report.independent_attempts}. Семейств песен: ${report.dataset.families}. Отражения не считаются отдельными играми.</p>`;
    for (const [split, label] of [['train', 'Обучение'], ['validation', 'Выбор модели'], ['test', 'Итоговая проверка']]) {
      const counts = report.dataset.splits[split];
      content += `<p>${label}: ${counts.attempts} попыток, ${counts.passes} проходов, ${counts.fails} провалов.</p>`;
    }
    content += '<div class="cnn-table-wrap"><table class="cnn-table"><thead><tr><th>Вариант</th><th>Ошибка accuracy, п.п.</th><th>Brier pass</th><th>Шаг обучения</th></tr></thead><tbody>';
    for (const [name, values] of Object.entries(report.variants)) {
      content += `<tr class="${Object.values(report.selected_on_validation).includes(name) ? 'cnn-selected' : ''}"><td>${escapeHtml(cnnVariants[name] || name)}</td><td>${values.test.mae_pp?.toFixed(2) ?? '-'}</td><td>${values.test.brier.toFixed(4)}</td><td>${values.best_step}</td></tr>`;
    }
    content += '</tbody></table></div>';
    for (const [head, label] of [['accuracy', 'Accuracy'], ['pass', 'Pass']]) {
      const release = report.release[head];
      const reason = release.reason === 'insufficient_validation' ? 'мало примеров в итоговой проверке' : 'улучшение не подтверждено';
      content += `<p><strong>${label}: ${release.accepted ? 'подключена к прогнозу' : 'не подключена'}</strong>. ${release.accepted ? 'Проверка подтвердила улучшение.' : reason + '.'} Выбран вариант: ${escapeHtml(cnnVariants[release.variant])}. Контроль: ${escapeHtml(cnnVariants[release.reference] || 'Модель по общим признакам')}.</p>`;
    }
    content += '<p>Варианты выбираются по промежуточной выборке. В таблице ошибки на отдельной итоговой выборке. Она не использовалась для выбора. Улучшение дополнительно проверяется повторными выборками целых семейств песен.</p>';
  } else if (!result.running) content += '<p>Обученной CNN для этого игрока пока нет. Приложение использует обычный расчет.</p>';
  $('#cnn-result').innerHTML = content;
  renderLocalCnnStatus(local);
  if (result.running || local.running) cnnPoll = setTimeout(() => action(loadCnnStatus), 3000);
}

const localTargets = {miss_or_combo_break:'Промах или срыв комбо', non_great_judgement:'Попадание ниже 300'};

function renderLocalCnnStatus(result) {
  const progress = result.progress || {};
  let content = '';
  if (result.running) {
    const label = progress.phase === 'replays' ? 'Проверяем события реплеев' : cnnPhases[progress.phase] || progress.phase;
    content += `<p>${escapeHtml(label)}: ${progress.step || 0}/${progress.total || 1}${progress.verified_replays !== undefined ? `, подтверждено: ${progress.verified_replays}` : ''}.</p><progress value="${Number(progress.step) || 0}" max="${Number(progress.total) || 1}"></progress>`;
  } else if (progress.status === 'failed' || progress.status === 'interrupted') {
    content += `<p>${escapeHtml(progress.error || progress.message || 'Обработка остановлена.')}</p>`;
  }
  const report = result.report;
  $('#local-cnn-report').hidden = !report;
  $('#local-cnn-report').href = `/api/users/${encodeURIComponent(state.user)}/cnn/local/report`;
  if (report) {
    content += `<p>Проверено полных реплеев: ${report.independent_attempts}. Это данные законченных карт, они не описывают все попытки игрока.</p>`;
    for (const [split,label] of [['train','Обучение'],['validation','Выбор модели'],['test','Итоговая проверка']]) {
      const s = report.dataset.splits[split];
      content += `<p>${label}: ${s.attempts} реплеев, ${s.families.length} семейств песен.</p>`;
    }
    content += '<div class="cnn-table-wrap"><table class="cnn-table"><thead><tr><th>Вариант</th><th>Brier: промах / комбо</th><th>Brier: ниже 300</th></tr></thead><tbody>';
    for (const [v,values] of Object.entries(report.variants)) {
      content += `<tr><td>${escapeHtml(cnnVariants[v] || v)}</td><td>${values.test.miss_or_combo_break.brier.toFixed(4)}</td><td>${values.test.non_great_judgement.brier.toFixed(4)}</td></tr>`;
    }
    content += `<tr><td>Контроль по признакам ноты</td><td>${report.baseline.test.miss_or_combo_break.brier.toFixed(4)}</td><td>${report.baseline.test.non_great_judgement.brier.toFixed(4)}</td></tr></tbody></table></div>`;
    for (const [name, release] of Object.entries(report.release)) {
      const note = release.accepted ? 'Улучшение подтверждено.' : release.reason === 'insufficient_validation' ? 'Мало независимых примеров проверки.' : release.reason === 'insufficient_mod_validation' ? 'Общее улучшение есть, но отдельные проверки модов его пока не подтверждают.' : 'Улучшение не подтверждено.';
      content += `<p><strong>${escapeHtml(localTargets[name])}: ${release.accepted ? 'подключено' : 'не подключено'}</strong>. ${note}</p>`;
    }
  } else if (!result.running) content += '<p>Обученной модели по участкам пока нет.</p>';
  $('#local-cnn-result').innerHTML = content;
}

function renderLocalPrediction(result) {
  const box = $('#local-patterns');
  if (!result || result.status === 'not_trained') { box.textContent=''; return; }
  if (result.status !== 'trained') {
    box.innerHTML = `<p class="api-hint">${result.status === 'rejected' ? 'Модель ошибок по участкам прошла обучение, но проверка не подтвердила качество для подключения.' : 'Для этой карты нет проверенного прогноза ошибок по участкам.'}</p>`;
    return;
  }
  const target = Object.keys(localTargets).find(k=>result.release[k]?.accepted);
  const peaks = [...result.windows].sort((a,b)=>b[target]-a[target]).slice(0,3);
  box.innerHTML = `<h3>Паттерны: прогноз по реплеям</h3><p class="api-hint">${escapeHtml(result.message)}</p>` + peaks.map(w=>`<p>${formatTime(w.start)}-${formatTime(w.end)}: ${Object.keys(localTargets).filter(k=>w[k]!==undefined).map(k=>`${escapeHtml(localTargets[k])}: ${w[k]}% нот`).join('; ')}</p>`).join('');
}

$('#cnn-button').addEventListener('click', () => action(async () => {
  if (!state.user) throw new Error('Сначала загрузи профиль игрока.');
  $('#cnn-dialog').showModal();
  await loadCnnStatus();
}));
$('#cnn-close').addEventListener('click', () => $('#cnn-dialog').close());
$('#cnn-dialog').addEventListener('close', () => clearTimeout(cnnPoll));
$('#cnn-refresh').addEventListener('click', () => action(loadCnnStatus));
$('#cnn-train').addEventListener('click', () => action(async () => {
  await api(`/api/users/${encodeURIComponent(state.user)}/cnn/train`, {method: 'POST'});
  await loadCnnStatus();
  toast('Обучение запущено. Прогноз переключится после успешной проверки.');
}, $('#cnn-train')));
$('#local-cnn-train').addEventListener('click', () => action(async () => {
  await api(`/api/users/${encodeURIComponent(state.user)}/cnn/local/train`, {method:'POST'});
  await loadCnnStatus();
}, $('#local-cnn-train')));
$('#cnn-stop').addEventListener('click', () => action(async () => {
  const result = await api(`/api/users/${encodeURIComponent(state.user)}/cnn/stop`, {method:'POST'});
  await loadCnnStatus();
  toast(result.stopped ? 'Обработка остановлена. Готовый кеш сохранен.' : 'Запущенной задачи этого игрока не найдено.');
}, $('#cnn-stop')));
$('#cnn-forward').addEventListener('click', () => action(async () => {
  const result = await api(`/api/users/${encodeURIComponent(state.user)}/cnn/forward`, {method:'POST'});
  await loadCnnStatus();
  toast(result.status === 'awaiting_new_plays' ? 'Пока нет попыток после сохранения модели.' : `Проверено новых попыток: ${result.attempts}. Модель не переобучалась.`);
}, $('#cnn-forward')));

$('#mods-picker').addEventListener('click', event => {
  const button = event.target.closest('button[data-mod]'); if (!button) return;
  const mod = button.dataset.mod;
  if (mod === 'NM') state.mods.clear();
  else if (state.mods.has(mod)) state.mods.delete(mod);
  else {
    if (mod === 'HR') state.mods.delete('EZ');
    if (mod === 'EZ') state.mods.delete('HR');
    if (mod === 'DT') state.mods.delete('HT');
    if (mod === 'HT') state.mods.delete('DT');
    state.mods.add(mod);
  }
  updateMods();
  action(async () => { if (state.mapId) await selectMap(state.mapId, false); await loadRecommendations(); });
});

$('#goal-tabs').addEventListener('click', event => {
  const button = event.target.closest('button[data-goal]'); if (!button) return;
  state.goal = button.dataset.goal;
  document.querySelectorAll('#goal-tabs button').forEach(item => item.classList.toggle('selected', item === button));
  action(loadRecommendations);
});

$('#log-play').addEventListener('click', () => {
  if (!state.user) { toast('Сначала укажи ник игрока.', true); $('#player-input').focus(); return; }
  $('#play-dialog').showModal();
});
$('#dialog-close').addEventListener('click', () => $('#play-dialog').close());
$('#play-form').addEventListener('submit', event => {
  event.preventDefault();
  action(async () => {
    await api('/api/plays', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({
      user: state.user, map_id: state.mapId, mods: currentMods(),
      accuracy: Number($('#play-accuracy').value), misses: Number($('#play-misses').value), passed: $('#play-passed').checked,
    }) });
    $('#play-dialog').close(); await loadSummary(); await selectMap(state.mapId, false); await loadRecommendations();
    toast('Результат добавлен в профиль.');
  });
});

$('#replay-file').addEventListener('change', event => {
  const file = event.target.files[0]; if (!file) return;
  action(async () => {
    const form = new FormData();
    form.append('user', state.user);
    form.append('passed', String($('#play-passed').checked));
    form.append('file', file);
    const result = await api('/api/plays/replay', { method: 'POST', body: form });
    $('#play-dialog').close();
    await loadSummary(); await selectMap(result.map_id, false); await loadRecommendations();
    toast(result.created ? `Реплей импортирован: ${result.accuracy}% точности.` : 'Этот реплей уже был импортирован.');
  });
  event.target.value = '';
});

async function init() {
  $('#player-input').value = state.user;
  try {
    const health = await api('/api/health');
    state.apiConfigured = health.api_configured;
    state.publicMode = health.public_mode;
    $('#oauth-local-note').hidden = !state.publicMode;
    for (const selector of ['#oauth-id', '#oauth-secret', '#oauth-submit', '#oauth-clear']) $(selector).disabled = !!state.publicMode;
    for (const selector of ['#cnn-button', '#local-import-button', '#nf-health-button', '#cookies-button', '#more-scores-button', '#local-options']) {
      $(selector).hidden = !!state.publicMode;
    }
    $('#api-hint').textContent = health.api_configured ? 'Официальный osu! API. До 100 разных карт recent и 100 best.' : 'Без входа: открытый API Hinamizawa. До 100 результатов recent и 100 best. Данные профиля передаются этому сервису.';
    await refresh();
    if (!state.publicMode) await refreshSession();
  } catch (error) { toast(error.message || 'Сервер недоступен.', true); }
}
init();
