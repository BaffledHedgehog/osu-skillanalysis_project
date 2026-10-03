const state = {
  user: localStorage.getItem('pulse-user') || '',
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
  return state.mods.size ? ['HD', 'HR', 'DT', 'EZ', 'HT'].filter(mod => state.mods.has(mod)).join('') : 'NM';
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
  list.innerHTML = state.maps.map(map => `<div class="catalog-item"><div><div class="catalog-name">${escapeHtml(map.artist)} — ${escapeHtml(map.title)}</div><div class="catalog-sub">${escapeHtml(map.version)} · ${escapeHtml(map.creator)} · ${formatTime(map.duration)}</div></div><button type="button" data-id="${escapeHtml(map.id)}">Разобрать →</button></div>`).join('');
  list.querySelectorAll('button').forEach(button => button.addEventListener('click', () => selectMap(button.dataset.id)));
}

async function loadSummary() {
  const box = $('#player-summary');
  if (!state.user) {
    box.innerHTML = '<span class="mini-dot"></span> Укажи ник, чтобы создать локальный профиль.';
    $('#avatar').textContent = '?';
    return;
  }
  $('#avatar').textContent = state.user[0].toUpperCase();
  const summary = await api(`/api/users/${encodeURIComponent(state.user)}/summary`);
  if (!summary.plays) {
    box.innerHTML = `<span class="mini-dot"></span><strong>${escapeHtml(state.user)}</strong> · Пока нет результатов. Добавь игру после анализа карты.`;
  } else {
    box.innerHTML = `<span class="mini-dot"></span><strong>${escapeHtml(state.user)}</strong> · ${summary.plays} результатов · ${summary.accuracy}% средняя точность${summary.comfort_stars ? ` · ориентир ${summary.comfort_stars}★` : ''}`;
  }
}

function renderPrediction(prediction) {
  const box = $('#prediction-content');
  if (!prediction || prediction.pass_probability === null) {
    const count = prediction?.observations || 0;
    box.innerHTML = `<div class="prediction-empty">${state.user ? `Есть ${count} из 3 необходимых результатов. После трёх игр появится первая персональная оценка.` : 'Укажи ник игрока и добавь минимум три результата. Тогда появится персональная оценка.'}<br><br>Сейчас можно изучать объективную нагрузку карты по участкам.</div>`;
    return;
  }
  const ml = prediction.ml_status;
  const mlNote = ml?.status === 'trained'
    ? `ML обучена на ${ml.plays} попытках и проверена на других картах (ошибка Brier: ${ml.metrics.brier}, базовый прогноз: ${ml.metrics.baseline_brier}).`
    : ml?.status === 'rejected'
      ? `ML пока не применяется: на отложенных картах её ошибка выше базового прогноза.`
      : `Для проверяемой ML модели нужны минимум 40 попыток на 10 разных картах, включая успехи и неудачи. Сейчас: ${ml?.plays || prediction.observations} попыток, ${ml?.maps || 0} карт.`;
  box.innerHTML = `<div class="prediction-row"><div><div class="prediction-label">ОЦЕНКА ШАНСА PASS</div><div class="prediction-number">${prediction.pass_probability}%</div></div><span class="confidence">${prediction.confidence === 'medium' ? 'Средняя уверенность' : 'Мало данных'}</span></div>
  <div class="prediction-label">ОЖИДАЕМАЯ ТОЧНОСТЬ</div><div class="accuracy-num">${prediction.expected_accuracy}% <span style="font-size:11px;color:#8995ab;font-weight:400">примерно ${prediction.accuracy_range[0]}–${prediction.accuracy_range[1]}%</span></div>
  <p class="prediction-message">По ${prediction.observations} результатам. ${escapeHtml(prediction.message)}<br>${escapeHtml(mlNote)}</p>
  <div style="margin-top:13px">${prediction.factors.map(factor => `<div class="prediction-factor">${escapeHtml(factor)}</div>`).join('')}</div>`;
}

function renderTimeline(analysis) {
  const windows = analysis.windows || [];
  const max = Math.max(1, ...windows.map(window => window.load));
  const bars = windows.map(window => {
    const total = Math.max(2, window.load / max * 100);
    const aim = window.load ? total * window.aim / window.load : 0;
    const speed = Math.max(0, total - aim);
    return `<div class="timeline-bar" title="${formatTime(window.start)} · Aim ${window.aim} · Speed ${window.speed}"><div class="aim-part" style="height:${aim}%"></div><div class="speed-part" style="height:${speed}%"></div></div>`;
  }).join('');
  $('#timeline').innerHTML = bars;
  const end = windows.length ? windows[windows.length - 1].end : analysis.duration;
  $('#timeline').insertAdjacentHTML('afterend', `<div class="timeline-axis" id="timeline-axis"><span>0:00</span><span>${formatTime(end / 2)}</span><span>${formatTime(end)}</span></div>`);
  document.querySelectorAll('#timeline-axis').forEach((axis, index) => { if (index > 0) axis.remove(); });
  $('#peak-list').innerHTML = analysis.peaks.map((peak, index) => `<div class="peak-item"><span class="peak-time">${formatTime(peak.start)}–${formatTime(peak.end)}</span><span>${index === 0 ? 'Максимальная нагрузка' : 'Сложный участок'}</span><span class="peak-type">${peak.aim >= peak.speed ? 'Aim' : 'Speed'}</span></div>`).join('');
}

function renderAnalysis(analysis) {
  state.analysis = analysis;
  $('#result-section').classList.remove('hidden');
  $('#map-difficulty').textContent = analysis.stars.toFixed(1);
  $('#map-title').textContent = `${analysis.artist} — ${analysis.title}`;
  $('#map-subtitle').textContent = `[${analysis.version}] · ${analysis.creator}`;
  $('#stat-stars').textContent = `${analysis.stars.toFixed(2)}★`;
  $('#stat-aim').textContent = analysis.aim.toFixed(2);
  $('#stat-speed').textContent = analysis.speed.toFixed(2);
  $('#stat-bpm').textContent = analysis.bpm;
  $('#stat-duration').textContent = `${formatTime(analysis.duration)} · ${analysis.object_count} объектов`;
  const official = $('#official-link');
  official.classList.toggle('hidden', !/^\d+$/.test(analysis.id));
  official.href = `https://osu.ppy.sh/beatmaps/${analysis.id}`;
  $('#dialog-map-name').textContent = `${analysis.artist} — ${analysis.title} [${analysis.version}] +${analysis.mods}`;
  renderPrediction(analysis.prediction);
  renderTimeline(analysis);
  updateMods();
}

async function selectMap(id, scroll = true) {
  state.mapId = id;
  localStorage.setItem('pulse-map', id);
  const query = new URLSearchParams({ user: state.user, mods: currentMods() });
  const analysis = await api(`/api/maps/${encodeURIComponent(id)}/analysis?${query}`);
  renderAnalysis(analysis);
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
  list.innerHTML = data.maps.map(map => `<article class="rec-card panel" data-id="${escapeHtml(map.id)}" tabindex="0" role="button" aria-label="Разобрать карту ${escapeHtml(map.title)}"><div class="rec-top"><span>osu!standard · +${escapeHtml(map.mods)}</span><span class="rec-stars">${map.stars.toFixed(2)}★</span></div><div class="rec-title">${escapeHtml(map.title)}</div><div class="rec-sub">${escapeHtml(map.artist)} · [${escapeHtml(map.version)}]</div><div class="rec-foot"><span>${formatTime(map.duration)} · ${map.bpm} BPM</span><span class="rec-chance">${map.prediction.pass_probability === null ? 'По звёздам' : `${map.prediction.pass_probability}% pass`}</span></div></article>`).join('');
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
  try {
    if (busyElement) { busyElement.disabled = true; busyElement.dataset.original = busyElement.textContent; busyElement.textContent = 'Подождите…'; }
    await fn();
  } catch (error) { toast(error.message || 'Произошла ошибка.', true); }
  finally { if (busyElement) { busyElement.disabled = false; busyElement.textContent = busyElement.dataset.original; } }
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
    state.user = $('#player-input').value.trim();
    localStorage.setItem('pulse-user', state.user);
    await loadSummary();
    if (state.mapId) await selectMap(state.mapId, false);
    await loadRecommendations();
    toast(`Профиль ${state.user} выбран.`);
  });
});

$('#import-button').addEventListener('click', () => action(async () => {
  if (!state.user) throw new Error('Сначала укажи ник игрока.');
  const result = await api('/api/users/import', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username: state.user }) });
  await refresh();
  toast(`Импортировано ${result.imported} результатов, новых карт: ${result.maps_fetched}.`);
}, $('#import-button')));

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
    $('#api-hint').textContent = health.api_configured ? 'Импортирует до 30 лучших и 30 недавних результатов по публичному профилю.' : 'Для импорта нужен OAuth ключ osu! в настройках сервера. Результаты можно добавлять вручную.';
    if (!health.api_configured) $('#import-button').disabled = true;
    await refresh();
  } catch (error) { toast(error.message || 'Сервер недоступен.', true); }
}
init();
