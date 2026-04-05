const dirSelect = document.getElementById('dirSelect');
const currentDirEl = document.getElementById('currentDir');

const widthMinInput = document.getElementById('widthMinInput');
const widthMaxInput = document.getElementById('widthMaxInput');
const heightMinInput = document.getElementById('heightMinInput');
const heightMaxInput = document.getElementById('heightMaxInput');
const areaMinInput = document.getElementById('areaMinInput');
const areaMaxInput = document.getElementById('areaMaxInput');

const includeRejectedToggle = document.getElementById('includeRejectedToggle');
const auditBtn = document.getElementById('auditBtn');
const selectAllBtn = document.getElementById('selectAllBtn');
const clearSelectionBtn = document.getElementById('clearSelectionBtn');
const pruneBtn = document.getElementById('pruneBtn');

const summaryMain = document.getElementById('summaryMain');
const summaryDetail = document.getElementById('summaryDetail');
const grid = document.getElementById('grid');

const state = {
    lastAuditConfig: null,
    lastStats: null,
    groupedByImage: new Map(),
    selectedImageIds: new Set(),
    baseDetailText: '',
};

function clamp01(value) {
    return Math.max(0, Math.min(1, value));
}

function formatPercent(value) {
    return `${(value * 100).toFixed(1)}%`;
}

function normalizeBound(raw, label) {
    const value = (raw || '').trim();
    if (!value) {
        return null;
    }

    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
        throw new Error(`${label} must be a number.`);
    }
    if (parsed < 0) {
        throw new Error(`${label} cannot be negative.`);
    }

    let normalized = parsed;
    if (normalized >= 1 && normalized <= 100) {
        normalized = normalized / 100;
    }
    if (normalized > 1) {
        throw new Error(`${label} must be between 0 and 1, or between 0 and 100.`);
    }
    return normalized;
}

function setInputToPercent(inputEl, value) {
    if (!inputEl) {
        return;
    }
    if (value === null || value === undefined) {
        inputEl.value = '';
        return;
    }
    inputEl.value = (value * 100).toFixed(2).replace(/\.00$/, '');
}

function describeMetricBound(name, min, max) {
    if (min === null && max === null) {
        return '';
    }
    if (min !== null && max !== null) {
        return `${name} ${formatPercent(min)}-${formatPercent(max)}`;
    }
    if (min !== null) {
        return `${name} >= ${formatPercent(min)}`;
    }
    return `${name} <= ${formatPercent(max)}`;
}

function describeFilters(config) {
    const parts = [
        describeMetricBound('width', config.widthMin, config.widthMax),
        describeMetricBound('height', config.heightMin, config.heightMax),
        describeMetricBound('area', config.areaMin, config.areaMax),
    ].filter(Boolean);
    return parts.length > 0 ? parts.join(' AND ') : '(no filters)';
}

function sameConfig(a, b) {
    if (!a || !b) {
        return false;
    }
    const keys = [
        'widthMin',
        'widthMax',
        'heightMin',
        'heightMax',
        'areaMin',
        'areaMax',
    ];
    for (const key of keys) {
        const av = a[key];
        const bv = b[key];
        if (av === null && bv === null) {
            continue;
        }
        if (av === null || bv === null) {
            return false;
        }
        if (Math.abs(Number(av) - Number(bv)) > 1e-9) {
            return false;
        }
    }
    return !!a.includeRejected === !!b.includeRejected;
}

function countSelectedBoxes() {
    let total = 0;
    for (const imageId of state.selectedImageIds) {
        const entry = state.groupedByImage.get(imageId);
        if (entry) {
            total += entry.boxes.length;
        }
    }
    return total;
}

function updateSelectionUi() {
    const selectedImages = state.selectedImageIds.size;
    const selectedBoxes = countSelectedBoxes();

    if (pruneBtn) {
        pruneBtn.textContent =
            selectedBoxes > 0
                ? `Prune Selected Boxes (${selectedBoxes})`
                : 'Prune Selected Boxes';
        pruneBtn.disabled = !state.lastAuditConfig || selectedBoxes === 0;
    }

    const detailParts = [];
    if (state.baseDetailText) {
        detailParts.push(state.baseDetailText);
    }
    if (state.groupedByImage.size > 0) {
        detailParts.push(`Selected images: ${selectedImages}`);
        detailParts.push(`Selected boxes: ${selectedBoxes}`);
    }
    summaryDetail.textContent = detailParts.join(' | ');
}

function setSummary(mainText, detailText = '') {
    summaryMain.innerHTML = mainText;
    state.baseDetailText = detailText;
    updateSelectionUi();
}

function syncTileSelectionState(imageId) {
    for (const tile of grid.querySelectorAll('.tile')) {
        if ((tile.dataset.imageId || '') !== imageId) {
            continue;
        }
        const selected = state.selectedImageIds.has(imageId);
        tile.classList.toggle('selected', selected);
        const checkbox = tile.querySelector('input[data-role="tile-select"]');
        if (checkbox) {
            checkbox.checked = selected;
        }
        return;
    }
}

function refreshAllTileSelectionStates() {
    for (const tile of grid.querySelectorAll('.tile')) {
        const imageId = tile.dataset.imageId || '';
        const selected = state.selectedImageIds.has(imageId);
        tile.classList.toggle('selected', selected);
        const checkbox = tile.querySelector('input[data-role="tile-select"]');
        if (checkbox) {
            checkbox.checked = selected;
        }
    }
}

function setImageSelection(imageId, selected) {
    if (selected) {
        state.selectedImageIds.add(imageId);
    } else {
        state.selectedImageIds.delete(imageId);
    }
    syncTileSelectionState(imageId);
    updateSelectionUi();
}

function clearSelection() {
    state.selectedImageIds.clear();
    refreshAllTileSelectionStates();
    updateSelectionUi();
}

function markStale() {
    state.lastAuditConfig = null;
    state.lastStats = null;
    state.groupedByImage = new Map();
    state.selectedImageIds.clear();
    if (pruneBtn) {
        pruneBtn.disabled = true;
        pruneBtn.textContent = 'Prune Selected Boxes';
    }
    setSummary('<strong>Stale:</strong> settings changed. Click Audit Grid.', '');
}

function groupMatchesByImage(matches) {
    const grouped = new Map();
    for (const hit of matches) {
        const imageId = hit.image_id;
        if (!grouped.has(imageId)) {
            grouped.set(imageId, {
                imageId,
                imageUrl:
                    hit.image_url ||
                    `/api/labeler/image_file/${encodeURIComponent(imageId)}`,
                boxes: [],
            });
        }
        grouped.get(imageId).boxes.push(hit);
    }
    return [...grouped.values()].sort((a, b) => b.boxes.length - a.boxes.length);
}

function addOverlayRect(svgEl, hit) {
    const xc = Number(hit.x_center);
    const yc = Number(hit.y_center);
    const w = Number(hit.width);
    const h = Number(hit.height);

    const x1 = xc - w / 2;
    const y1 = yc - h / 2;
    const x2 = xc + w / 2;
    const y2 = yc + h / 2;

    const cx1 = clamp01(x1);
    const cy1 = clamp01(y1);
    const cx2 = clamp01(x2);
    const cy2 = clamp01(y2);

    const rw = cx2 - cx1;
    const rh = cy2 - cy1;
    if (rw <= 0 || rh <= 0) {
        return;
    }

    const ns = 'http://www.w3.org/2000/svg';
    const rect = document.createElementNS(ns, 'rect');
    rect.setAttribute('x', String(cx1));
    rect.setAttribute('y', String(cy1));
    rect.setAttribute('width', String(rw));
    rect.setAttribute('height', String(rh));
    rect.setAttribute('fill', 'rgba(239, 68, 68, 0.18)');
    rect.setAttribute('stroke', 'rgba(248, 113, 113, 0.95)');
    rect.setAttribute('stroke-width', '0.004');
    svgEl.appendChild(rect);
}

function renderGrid(matches) {
    grid.innerHTML = '';

    const grouped = groupMatchesByImage(matches);
    state.groupedByImage = new Map(grouped.map((item) => [item.imageId, item]));

    if (grouped.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'empty';
        empty.textContent = 'No oversized boxes found for current settings.';
        grid.appendChild(empty);
        updateSelectionUi();
        return;
    }

    for (const item of grouped) {
        const tile = document.createElement('div');
        tile.className = 'tile';
        tile.dataset.imageId = item.imageId;
        if (state.selectedImageIds.has(item.imageId)) {
            tile.classList.add('selected');
        }

        const selectChip = document.createElement('label');
        selectChip.className = 'select-chip';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.dataset.role = 'tile-select';
        checkbox.checked = state.selectedImageIds.has(item.imageId);
        checkbox.addEventListener('click', (ev) => {
            ev.stopPropagation();
        });
        checkbox.addEventListener('change', () => {
            setImageSelection(item.imageId, checkbox.checked);
        });

        const chipText = document.createElement('span');
        chipText.textContent = 'Select';

        selectChip.appendChild(checkbox);
        selectChip.appendChild(chipText);

        const imageWrap = document.createElement('div');
        imageWrap.className = 'image-wrap';

        const img = document.createElement('img');
        img.loading = 'lazy';
        img.src = item.imageUrl;
        img.alt = item.imageId;

        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('class', 'overlay');
        svg.setAttribute('viewBox', '0 0 1 1');
        svg.setAttribute('preserveAspectRatio', 'none');

        for (const hit of item.boxes) {
            addOverlayRect(svg, hit);
        }

        const badge = document.createElement('div');
        badge.className = 'badge';
        badge.textContent = `${item.boxes.length} box${item.boxes.length === 1 ? '' : 'es'}`;

        imageWrap.appendChild(img);
        imageWrap.appendChild(svg);
        imageWrap.appendChild(selectChip);
        imageWrap.appendChild(badge);

        const meta = document.createElement('div');
        meta.className = 'meta';

        const idLine = document.createElement('div');
        idLine.className = 'id';
        idLine.textContent = item.imageId;

        const maxWidth = Math.max(...item.boxes.map((hit) => Number(hit.width) || 0));
        const maxHeight = Math.max(...item.boxes.map((hit) => Number(hit.height) || 0));
        const maxArea = Math.max(...item.boxes.map((hit) => Number(hit.area) || 0));
        const statLine = document.createElement('div');
        statLine.textContent = `Max w=${formatPercent(maxWidth)} h=${formatPercent(maxHeight)} area=${formatPercent(maxArea)}`;

        const actions = document.createElement('div');
        actions.className = 'actions';

        const imgLink = document.createElement('a');
        imgLink.href = item.imageUrl;
        imgLink.target = '_blank';
        imgLink.rel = 'noopener';
        imgLink.textContent = 'Open Image';

        const labelerLink = document.createElement('a');
        labelerLink.href = `/labeler?image_id=${encodeURIComponent(item.imageId)}`;
        labelerLink.target = '_blank';
        labelerLink.rel = 'noopener';
        labelerLink.textContent = 'Open in Labeler';

        actions.appendChild(imgLink);
        actions.appendChild(labelerLink);

        meta.appendChild(idLine);
        meta.appendChild(statLine);
        meta.appendChild(actions);

        tile.appendChild(imageWrap);
        tile.appendChild(meta);
        tile.addEventListener('click', (ev) => {
            if (ev.target.closest('a') || ev.target.closest('.select-chip')) {
                return;
            }
            const nextSelected = !state.selectedImageIds.has(item.imageId);
            setImageSelection(item.imageId, nextSelected);
        });

        grid.appendChild(tile);
    }

    updateSelectionUi();
}

function readFilterConfigFromInputs() {
    const config = {
        widthMin: normalizeBound(widthMinInput.value, 'Width min'),
        widthMax: normalizeBound(widthMaxInput.value, 'Width max'),
        heightMin: normalizeBound(heightMinInput.value, 'Height min'),
        heightMax: normalizeBound(heightMaxInput.value, 'Height max'),
        areaMin: normalizeBound(areaMinInput.value, 'Area min'),
        areaMax: normalizeBound(areaMaxInput.value, 'Area max'),
        includeRejected: !!includeRejectedToggle.checked,
    };

    const pairs = [
        ['Width', config.widthMin, config.widthMax],
        ['Height', config.heightMin, config.heightMax],
        ['Area', config.areaMin, config.areaMax],
    ];

    let activeBounds = 0;
    for (const [name, min, max] of pairs) {
        if (min !== null) activeBounds += 1;
        if (max !== null) activeBounds += 1;
        if (min !== null && max !== null && min > max) {
            throw new Error(`${name} min cannot be greater than ${name} max.`);
        }
    }
    if (activeBounds === 0) {
        throw new Error('Set at least one filter bound (min or max).');
    }

    setInputToPercent(widthMinInput, config.widthMin);
    setInputToPercent(widthMaxInput, config.widthMax);
    setInputToPercent(heightMinInput, config.heightMin);
    setInputToPercent(heightMaxInput, config.heightMax);
    setInputToPercent(areaMinInput, config.areaMin);
    setInputToPercent(areaMaxInput, config.areaMax);

    return config;
}

function makeAuditRequestPayload(config) {
    return {
        width_min: config.widthMin,
        width_max: config.widthMax,
        height_min: config.heightMin,
        height_max: config.heightMax,
        area_min: config.areaMin,
        area_max: config.areaMax,
        include_rejected: config.includeRejected,
        max_results: 5000,
    };
}

function makePruneSelectedRequestPayload(config, targets) {
    return {
        targets,
        width_min: config.widthMin,
        width_max: config.widthMax,
        height_min: config.heightMin,
        height_max: config.heightMax,
        area_min: config.areaMin,
        area_max: config.areaMax,
        include_rejected: config.includeRejected,
        dry_run: false,
    };
}

async function runAudit() {
    let config;
    try {
        config = readFilterConfigFromInputs();
    } catch (err) {
        setSummary(`<strong>Invalid settings:</strong> ${err.message || err}`, '');
        return;
    }

    auditBtn.disabled = true;
    pruneBtn.disabled = true;
    setSummary(
        '<strong>Auditing...</strong>',
        `Filters: ${describeFilters(config)}`
    );

    try {
        const resp = await fetch('/api/labeler/oversized-boxes/audit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(makeAuditRequestPayload(config)),
        });
        if (!resp.ok) {
            const payload = await resp.json().catch(() => ({}));
            throw new Error(payload.detail || `Audit failed (HTTP ${resp.status})`);
        }

        const payload = await resp.json();
        const stats = payload.stats || {};
        const matches = payload.matches || [];
        const filterBounds = stats.filter_bounds || {};

        state.lastAuditConfig = {
            widthMin: filterBounds.width_min ?? config.widthMin,
            widthMax: filterBounds.width_max ?? config.widthMax,
            heightMin: filterBounds.height_min ?? config.heightMin,
            heightMax: filterBounds.height_max ?? config.heightMax,
            areaMin: filterBounds.area_min ?? config.areaMin,
            areaMax: filterBounds.area_max ?? config.areaMax,
            includeRejected: config.includeRejected,
        };
        state.lastStats = stats;
        state.selectedImageIds.clear();

        const found = Number(stats.oversized_boxes_found || 0);
        const imagesWith = Number(stats.images_with_oversized_boxes || 0);
        const returned = Number(stats.matches_returned || matches.length);
        const trunc = !!stats.matches_truncated;

        const detail = [
            `Filters: ${describeFilters(state.lastAuditConfig)}`,
            `Synthetic considered: ${stats.synthetic_images_total || 0}`,
            `Rejected excluded: ${stats.rejected_synthetic_excluded || 0}`,
            `Returned: ${returned}${trunc ? '+' : ''}`,
        ].join(' | ');

        setSummary(
            `<strong>Audit results:</strong> ${found} box matches across ${imagesWith} images`,
            detail
        );

        renderGrid(matches);
    } catch (err) {
        state.lastAuditConfig = null;
        state.lastStats = null;
        state.groupedByImage = new Map();
        state.selectedImageIds.clear();
        grid.innerHTML = '';
        setSummary(`<strong>Audit failed:</strong> ${err.message || err}`, '');
    } finally {
        auditBtn.disabled = false;
        updateSelectionUi();
    }
}

function buildSelectedTargets() {
    const targets = [];
    for (const imageId of state.selectedImageIds) {
        const entry = state.groupedByImage.get(imageId);
        if (!entry) {
            continue;
        }
        const boxKeys = entry.boxes
            .map((hit) => String(hit.box_key || '').trim())
            .filter(Boolean);
        if (boxKeys.length === 0) {
            continue;
        }
        targets.push({ image_id: imageId, box_keys: boxKeys });
    }
    return targets;
}

async function runPruneSelected() {
    let currentConfig;
    try {
        currentConfig = readFilterConfigFromInputs();
    } catch (err) {
        setSummary(`<strong>Invalid settings:</strong> ${err.message || err}`, '');
        return;
    }

    if (!sameConfig(state.lastAuditConfig, currentConfig)) {
        setSummary(
            '<strong>Run audit first:</strong> prune only runs against the currently audited filter settings.',
            ''
        );
        return;
    }

    const selectedImages = state.selectedImageIds.size;
    const selectedBoxes = countSelectedBoxes();
    if (selectedImages === 0 || selectedBoxes === 0) {
        setSummary('<strong>No selection:</strong> select one or more audited images first.', '');
        return;
    }

    const targets = buildSelectedTargets();
    if (targets.length === 0) {
        setSummary('<strong>No prune targets:</strong> selected images have no audited box keys.', '');
        return;
    }

    const ok = window.confirm(
        `Prune ${selectedBoxes} audited boxes across ${selectedImages} selected images?\n\nFilters: ${describeFilters(currentConfig)}\n\nOnly matching bbox lines are removed. Images are not deleted, and non-target boxes are untouched.`
    );
    if (!ok) {
        return;
    }

    pruneBtn.disabled = true;
    auditBtn.disabled = true;
    setSummary('<strong>Pruning selected...</strong>', 'Updating selected label files only.');

    try {
        const resp = await fetch('/api/labeler/oversized-boxes/prune-selected', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(makePruneSelectedRequestPayload(currentConfig, targets)),
        });
        if (!resp.ok) {
            const payload = await resp.json().catch(() => ({}));
            throw new Error(payload.detail || `Prune failed (HTTP ${resp.status})`);
        }

        const payload = await resp.json();
        const stats = payload.stats || {};
        alert([
            `Run: ${stats.run_dir || '-'}`,
            `Selected images targeted: ${stats.images_targeted || 0}`,
            `Box keys targeted: ${stats.boxes_targeted || 0}`,
            `Boxes removed: ${stats.boxes_removed || 0}`,
            `Images modified: ${stats.images_modified || 0}`,
            `Requested boxes not found: ${stats.boxes_requested_not_found || 0}`,
        ].join('\n'));

        await runAudit();
    } catch (err) {
        setSummary(`<strong>Prune failed:</strong> ${err.message || err}`, '');
    } finally {
        auditBtn.disabled = false;
        updateSelectionUi();
    }
}

async function loadDirectories() {
    try {
        const [dirsRes, currentRes] = await Promise.all([
            fetch('/api/directories/list'),
            fetch('/api/directories/current'),
        ]);
        if (!dirsRes.ok || !currentRes.ok) {
            throw new Error('Directory APIs unavailable');
        }

        const dirs = await dirsRes.json();
        const current = await currentRes.json();

        dirSelect.innerHTML = '';
        for (const d of dirs) {
            const opt = document.createElement('option');
            opt.value = d.path;
            opt.textContent = d.label;
            dirSelect.appendChild(opt);
        }

        if (current && current.path) {
            dirSelect.value = current.path;
            currentDirEl.textContent = `(current: ${current.name})`;
            currentDirEl.title = current.path;
        }
    } catch (err) {
        console.error(err);
        dirSelect.innerHTML = '<option value="">Directory API unavailable</option>';
        dirSelect.disabled = true;
        setSummary('<strong>Error:</strong> unable to load directories.', '');
    }
}

async function setDirectory(path) {
    const resp = await fetch('/api/directories/set', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
    });
    if (!resp.ok) {
        const payload = await resp.json().catch(() => ({}));
        throw new Error(payload.detail || `Directory switch failed (HTTP ${resp.status})`);
    }
    return resp.json();
}

dirSelect.addEventListener('change', async () => {
    if (!dirSelect.value) {
        return;
    }

    try {
        dirSelect.disabled = true;
        const result = await setDirectory(dirSelect.value);
        currentDirEl.textContent = `(current: ${result.name})`;
        currentDirEl.title = result.path;
        markStale();
        await runAudit();
    } catch (err) {
        setSummary(`<strong>Directory error:</strong> ${err.message || err}`, '');
    } finally {
        dirSelect.disabled = false;
    }
});

auditBtn.addEventListener('click', async () => {
    await runAudit();
});

selectAllBtn.addEventListener('click', () => {
    for (const imageId of state.groupedByImage.keys()) {
        state.selectedImageIds.add(imageId);
    }
    refreshAllTileSelectionStates();
    updateSelectionUi();
});

clearSelectionBtn.addEventListener('click', () => {
    clearSelection();
});

pruneBtn.addEventListener('click', async () => {
    await runPruneSelected();
});

widthMinInput.addEventListener('input', markStale);
widthMaxInput.addEventListener('input', markStale);
heightMinInput.addEventListener('input', markStale);
heightMaxInput.addEventListener('input', markStale);
areaMinInput.addEventListener('input', markStale);
areaMaxInput.addEventListener('input', markStale);
includeRejectedToggle.addEventListener('change', markStale);

async function initialize() {
    await loadDirectories();
    await runAudit();
}

initialize();
