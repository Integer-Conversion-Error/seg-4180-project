const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let images = [];
let currentIndex = 0;
let currentImage = null;
let imageData = null;
let imgElement = null;
let autosaveOnNavigate = false;
let currentDirectoryPath = null;
let selectedBoxIndex = -1;
let boxesDirty = false;
let lookalikeSimilarityByBoxKey = {};
let zoomLevel = 1;

const MIN_ZOOM = 0.25;
const MAX_ZOOM = 4;
const ZOOM_STEP = 0.25;

const dirSelect = document.getElementById('dirSelect');
const currentDirLabel = document.getElementById('currentDir');
const zoomSlider = document.getElementById('zoomSlider');
const zoomLevelLabel = document.getElementById('zoomLevel');

const CLASS_COLORS = {
    0: '#4ade80',
    1: '#f59e0b',
    2: '#f97316',
    3: '#ef4444',
    4: '#a78bfa'
};

const CLASS_NAMES = {
    0: 'vehicle',
    1: 'enforcement_vehicle',
    2: 'police_old',
    3: 'police_new',
    4: 'lookalike_negative'
};

function clamp01(value) {
    return Math.max(0, Math.min(1, value));
}

function clampZoom(value) {
    return Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, value));
}

function applyZoom() {
    if (zoomSlider) {
        zoomSlider.value = String(Math.round(zoomLevel * 100));
    }
    if (zoomLevelLabel) {
        zoomLevelLabel.textContent = `${Math.round(zoomLevel * 100)}%`;
    }
    if (!canvas.width || !canvas.height) {
        return;
    }
    canvas.style.width = `${Math.round(canvas.width * zoomLevel)}px`;
    canvas.style.height = `${Math.round(canvas.height * zoomLevel)}px`;
}

function setZoom(nextZoom) {
    zoomLevel = clampZoom(nextZoom);
    applyZoom();
}

function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const displayWidth = canvas.clientWidth || rect.width;
    const displayHeight = canvas.clientHeight || rect.height;
    const scaleX = canvas.width / displayWidth;
    const scaleY = canvas.height / displayHeight;
    return {
        x: (e.clientX - rect.left - canvas.clientLeft) * scaleX,
        y: (e.clientY - rect.top - canvas.clientTop) * scaleY,
    };
}

function normalizedBoxKeyForBox(box, forcedCls = null) {
    const imgW = canvas.width || imgElement?.naturalWidth || 1;
    const imgH = canvas.height || imgElement?.naturalHeight || 1;
    const cls = forcedCls !== null ? forcedCls : Number(box.cls);

    const x1 = clamp01(Number(box.x1) / imgW);
    const y1 = clamp01(Number(box.y1) / imgH);
    const x2 = clamp01(Number(box.x2) / imgW);
    const y2 = clamp01(Number(box.y2) / imgH);

    const xCenter = (x1 + x2) / 2;
    const yCenter = (y1 + y2) / 2;
    const width = Math.max(0, x2 - x1);
    const height = Math.max(0, y2 - y1);
    return `${cls}:${xCenter.toFixed(6)}:${yCenter.toFixed(6)}:${width.toFixed(6)}:${height.toFixed(6)}`;
}

function getCurrentLookalikeBoxes() {
    if (!imageData || !Array.isArray(imageData.boxes)) {
        return [];
    }

    const entries = [];
    let lookalikeIndex = 0;
    for (let i = 0; i < imageData.boxes.length; i++) {
        const box = imageData.boxes[i];
        if (Number(box.cls) !== 4) {
            continue;
        }
        const boxKey = normalizedBoxKeyForBox(box, 4);
        entries.push({
            index: lookalikeIndex,
            source_index: i,
            box_key: boxKey,
            x1: box.x1,
            y1: box.y1,
            x2: box.x2,
            y2: box.y2,
            similar_to: lookalikeSimilarityByBoxKey[boxKey] || '',
        });
        lookalikeIndex += 1;
    }
    return entries;
}

function updateLookalikeCount() {
    document.getElementById('lookalikeCount').textContent = String(getCurrentLookalikeBoxes().length);
}

function findBoxIndexAtCoords(coords) {
    if (!imageData || !Array.isArray(imageData.boxes)) {
        return -1;
    }
    for (let i = imageData.boxes.length - 1; i >= 0; i--) {
        const box = imageData.boxes[i];
        if (coords.x >= box.x1 && coords.x <= box.x2 && coords.y >= box.y1 && coords.y <= box.y2) {
            return i;
        }
    }
    return -1;
}

function setSelectedBoxClass(newCls) {
    if (!imageData || !Array.isArray(imageData.boxes) || selectedBoxIndex < 0) {
        return;
    }
    const box = imageData.boxes[selectedBoxIndex];
    const prevCls = Number(box.cls);
    if (prevCls === newCls) {
        return;
    }

    if (prevCls === 4) {
        const oldKey = normalizedBoxKeyForBox(box, 4);
        delete lookalikeSimilarityByBoxKey[oldKey];
    }

    box.cls = newCls;

    if (newCls === 4) {
        const newKey = normalizedBoxKeyForBox(box, 4);
        if (!(newKey in lookalikeSimilarityByBoxKey)) {
            lookalikeSimilarityByBoxKey[newKey] = '';
        }
    }

    boxesDirty = true;
    render();
    renderSimilarityControls();
    updateLookalikeCount();
}

function deleteBoxAtIndex(index) {
    if (!imageData || !Array.isArray(imageData.boxes) || index < 0 || index >= imageData.boxes.length) {
        return;
    }
    const box = imageData.boxes[index];
    if (Number(box.cls) === 4) {
        const key = normalizedBoxKeyForBox(box, 4);
        delete lookalikeSimilarityByBoxKey[key];
    }

    imageData.boxes.splice(index, 1);
    if (selectedBoxIndex === index) {
        selectedBoxIndex = -1;
    } else if (selectedBoxIndex > index) {
        selectedBoxIndex -= 1;
    }

    boxesDirty = true;
    render();
    renderSimilarityControls();
    updateLookalikeCount();
}

async function loadWorkingDirectories() {
    if (!dirSelect) {
        return;
    }

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
            currentDirectoryPath = current.path;
            dirSelect.value = current.path;
            if (currentDirLabel) {
                currentDirLabel.textContent = `(current: ${current.name})`;
                currentDirLabel.title = current.path;
            }
        }
    } catch (err) {
        console.error('Failed to load directories:', err);
        dirSelect.innerHTML = '<option value="">Directory API unavailable</option>';
        dirSelect.disabled = true;
        if (currentDirLabel) {
            currentDirLabel.textContent = '';
        }
    }
}

async function setWorkingDirectory(path) {
    const resp = await fetch('/api/directories/set', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
    });

    if (!resp.ok) {
        throw new Error(`Failed to set directory (HTTP ${resp.status})`);
    }

    return resp.json();
}

async function loadImages() {
    const resp = await fetch('/api/lookalike/images');
    if (!resp.ok) {
        throw new Error(`Failed to load lookalike images (HTTP ${resp.status})`);
    }
    images = await resp.json();

    if (images.length === 0) {
        currentImage = null;
        imageData = null;
        selectedBoxIndex = -1;
        boxesDirty = false;
        lookalikeSimilarityByBoxKey = {};
        render();
        renderSimilarityControls();
        document.getElementById('imageId').textContent = '-';
        document.getElementById('imageStatus').textContent = '-';
        document.getElementById('expectedClass').textContent = '-';
        document.getElementById('lookalikeCount').textContent = '0';
        document.getElementById('progress').textContent = '0 / 0';
        document.getElementById('hint').textContent = 'No images with class-4 boxes found.';
        return;
    }

    await loadImage(0);
}

async function loadImage(index) {
    if (index < 0 || index >= images.length) {
        return;
    }

    currentIndex = index;
    currentImage = images[index];

    const resp = await fetch(`/api/lookalike/image/${currentImage.image_id}`);
    if (!resp.ok) {
        throw new Error(`Failed to load image details (HTTP ${resp.status})`);
    }
    imageData = await resp.json();
    selectedBoxIndex = -1;
    boxesDirty = false;
    lookalikeSimilarityByBoxKey = {};
    for (const entry of imageData.lookalike_boxes || []) {
        if (entry.box_key) {
            lookalikeSimilarityByBoxKey[entry.box_key] = entry.similar_to || '';
        }
    }

    document.getElementById('imageId').textContent = imageData.image_id.substring(0, 16) + '...';
    document.getElementById('imageStatus').textContent = imageData.is_synthetic ? 'Synthetic' : 'Original';
    document.getElementById('expectedClass').textContent = imageData.expected_class || 'none';
    updateLookalikeCount();
    document.getElementById('progress').textContent = `${currentIndex + 1} / ${images.length}`;

    if (!imgElement) {
        imgElement = new Image();
    }
    imgElement.onload = () => {
        canvas.width = imgElement.naturalWidth;
        canvas.height = imgElement.naturalHeight;
        applyZoom();
        render();
        renderSimilarityControls();
    };
    imgElement.src = imageData.image_path;

    document.getElementById('hint').textContent = 'Select a box to edit class/tags. Right-click deletes; middle-click sets vehicle. Save to persist.';
}

function render() {
    if (!imgElement || !imageData) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgElement, 0, 0);

    const boxes = imageData.boxes || [];
    boxes.forEach((box, idx) => {
        const cls = Number(box.cls);
        const color = CLASS_COLORS[cls] || '#fff';
        ctx.strokeStyle = color;
        ctx.lineWidth = cls === 4 ? 3 : 2;
        ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);

        if (idx === selectedBoxIndex) {
            ctx.save();
            ctx.strokeStyle = '#93c5fd';
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 4]);
            ctx.strokeRect(box.x1 - 2, box.y1 - 2, box.x2 - box.x1 + 4, box.y2 - box.y1 + 4);
            ctx.restore();
        }

        ctx.fillStyle = color;
        ctx.font = '14px sans-serif';
        ctx.fillText(CLASS_NAMES[cls] || `class_${cls}`, box.x1, Math.max(12, box.y1 - 4));
    });
}

function renderSimilarityControls() {
    const list = document.getElementById('boxMetaList');
    list.innerHTML = '';

    const lookalikeBoxes = getCurrentLookalikeBoxes();
    if (!imageData || lookalikeBoxes.length === 0) {
        list.innerHTML = '<p style="color:#777; font-size:13px;">No class-4 boxes for this image.</p>';
        return;
    }

    const options = imageData.similarity_options || [];
    for (const box of lookalikeBoxes) {
        const container = document.createElement('div');
        container.className = 'box-meta-item';
        if (box.source_index === selectedBoxIndex) {
            container.style.outline = '2px solid #60a5fa';
        }
        container.onclick = () => {
            selectedBoxIndex = box.source_index;
            render();
            renderSimilarityControls();
        };

        const label = document.createElement('label');
        label.textContent = `Lookalike box #${box.index + 1}`;
        container.appendChild(label);

        const select = document.createElement('select');
        select.dataset.boxKey = box.box_key;
        const blank = document.createElement('option');
        blank.value = '';
        blank.textContent = '-- not tagged --';
        select.appendChild(blank);

        for (const option of options) {
            const el = document.createElement('option');
            el.value = option;
            el.textContent = option;
            select.appendChild(el);
        }
        select.value = box.similar_to || '';
        select.onchange = () => {
            lookalikeSimilarityByBoxKey[box.box_key] = select.value || '';
            if (!select.value) {
                delete lookalikeSimilarityByBoxKey[box.box_key];
            }
            const hint = document.getElementById('hint');
            if (hint) {
                hint.textContent = 'Similarity tags changed. Save to persist.';
            }
        };
        container.appendChild(select);

        const coords = document.createElement('div');
        coords.className = 'coords';
        coords.textContent = `${Math.round(box.x1)},${Math.round(box.y1)} - ${Math.round(box.x2)},${Math.round(box.y2)}`;
        container.appendChild(coords);

        list.appendChild(container);
    }
}

async function parseApiError(resp, prefix) {
    let message = '';
    try {
        const body = await resp.json();
        const detail = body && body.detail;
        if (detail && typeof detail === 'object' && detail.message) {
            message = String(detail.message);
        } else if (typeof detail === 'string') {
            message = detail;
        } else {
            message = JSON.stringify(body);
        }
    } catch (_err) {
        message = await resp.text();
    }

    const error = new Error(`${prefix} (HTTP ${resp.status}): ${message}`);
    error.statusCode = resp.status;
    return error;
}

async function saveCurrentLabelsIfNeeded() {
    if (!currentImage || !imageData || !boxesDirty) {
        return { saved: false, boxes: 0 };
    }

    const imgW = canvas.width || imgElement?.naturalWidth || 1;
    const imgH = canvas.height || imgElement?.naturalHeight || 1;
    const payloadBoxes = (imageData.boxes || []).map((box) => ({
        x1: clamp01(Number(box.x1) / imgW),
        y1: clamp01(Number(box.y1) / imgH),
        x2: clamp01(Number(box.x2) / imgW),
        y2: clamp01(Number(box.y2) / imgH),
        cls: Number(box.cls),
    }));

    const resp = await fetch(`/api/labeler/labels/${currentImage.image_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ boxes: payloadBoxes }),
    });
    if (!resp.ok) {
        throw await parseApiError(resp, 'Label save failed');
    }

    const out = await resp.json();
    boxesDirty = false;
    return {
        saved: true,
        boxes: Number(out.boxes || 0),
    };
}

function buildLookalikeMetadataEntries() {
    const lookalikeBoxes = getCurrentLookalikeBoxes();
    return lookalikeBoxes.map((box) => ({
        box_key: box.box_key,
        similar_to: lookalikeSimilarityByBoxKey[box.box_key] || '',
    }));
}

async function saveCurrentMetadata(showMessage = true) {
    if (!currentImage || !imageData) {
        return;
    }

    const labelResult = await saveCurrentLabelsIfNeeded();
    const entries = buildLookalikeMetadataEntries();

    let metadataSaved = false;
    if (entries.length > 0) {
        const resp = await fetch(`/api/lookalike/metadata/${currentImage.image_id}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ entries }),
        });
        if (!resp.ok) {
            throw await parseApiError(resp, 'Metadata save failed');
        }
        metadataSaved = true;
    }

    if (showMessage) {
        let message = 'Nothing to save.';
        if (labelResult.saved && metadataSaved) {
            message = 'Saved labels and lookalike metadata.';
        } else if (labelResult.saved) {
            message = 'Saved labels. No lookalike boxes remain for metadata tags.';
        } else if (metadataSaved) {
            message = 'Saved lookalike metadata.';
        }

        document.getElementById('hint').textContent = message;
        setTimeout(() => {
            if (document.getElementById('hint').textContent === message) {
                document.getElementById('hint').textContent = 'Select a box to edit class/tags. Right-click deletes; middle-click sets vehicle. Save to persist.';
            }
        }, 1400);
    }
}

async function navigateTo(index) {
    if (index < 0 || index >= images.length) {
        return;
    }
    try {
        if (autosaveOnNavigate && currentImage) {
            await saveCurrentMetadata(false);
        }
        await loadImage(index);
    } catch (err) {
        console.error(err);
        if (err && err.statusCode === 409 && currentImage) {
            await loadImage(currentIndex);
            document.getElementById('hint').textContent = 'Boxes changed. Refreshed current image. Re-apply tags and save.';
            return;
        }
        document.getElementById('hint').textContent = `Navigation blocked: ${err.message || err}`;
    }
}

canvas.addEventListener('mousedown', (e) => {
    if (e.button === 1) {
        e.preventDefault();
        return;
    }
    if (e.button !== 0) {
        return;
    }
    if (!imageData) {
        return;
    }

    const coords = getCanvasCoords(e);
    selectedBoxIndex = findBoxIndexAtCoords(coords);
    render();
    renderSimilarityControls();
});

canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    if (!imageData) {
        return;
    }

    const coords = getCanvasCoords(e);
    const hitIndex = findBoxIndexAtCoords(coords);
    if (hitIndex < 0) {
        return;
    }

    deleteBoxAtIndex(hitIndex);
    document.getElementById('hint').textContent = 'Deleted box via right-click. Save to persist.';
});

canvas.addEventListener('auxclick', (e) => {
    if (e.button !== 1) {
        return;
    }
    e.preventDefault();
    if (!imageData) {
        return;
    }

    const coords = getCanvasCoords(e);
    const hitIndex = findBoxIndexAtCoords(coords);
    if (hitIndex < 0) {
        return;
    }

    selectedBoxIndex = hitIndex;
    setSelectedBoxClass(0);
    document.getElementById('hint').textContent = 'Set box class to vehicle via middle-click. Save to persist.';
});

document.getElementById('prevBtn').onclick = () => navigateTo(currentIndex - 1);
document.getElementById('nextBtn').onclick = () => navigateTo(currentIndex + 1);
document.getElementById('saveBtn').onclick = async () => {
    try {
        await saveCurrentMetadata(true);
    } catch (err) {
        console.error(err);
        if (err && err.statusCode === 409 && currentImage) {
            await loadImage(currentIndex);
            document.getElementById('hint').textContent = 'Boxes changed. Refreshed current image. Re-apply tags and save.';
            return;
        }
        document.getElementById('hint').textContent = `Save failed: ${err.message || err}`;
    }
};

document.getElementById('autosaveToggle').onchange = (e) => {
    autosaveOnNavigate = !!e.target.checked;
    localStorage.setItem('lookalike_autosave_on_navigate', autosaveOnNavigate ? '1' : '0');
};

const zoomOutBtn = document.getElementById('zoomOutBtn');
if (zoomOutBtn) {
    zoomOutBtn.onclick = () => setZoom(zoomLevel - ZOOM_STEP);
}

const zoomInBtn = document.getElementById('zoomInBtn');
if (zoomInBtn) {
    zoomInBtn.onclick = () => setZoom(zoomLevel + ZOOM_STEP);
}

const zoomResetBtn = document.getElementById('zoomResetBtn');
if (zoomResetBtn) {
    zoomResetBtn.onclick = () => setZoom(1);
}

if (zoomSlider) {
    zoomSlider.addEventListener('input', (e) => {
        const next = Number(e.target.value) / 100;
        setZoom(next);
    });
}

function handleShortcut(e) {
    const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : '';
    if (tag === 'input' || tag === 'textarea' || tag === 'select' || e.target?.isContentEditable) {
        return;
    }

    const key = (e.key || '').toLowerCase();
    if (key === 'arrowright' || key === 'n' || key === 'd') {
        e.preventDefault();
        navigateTo(currentIndex + 1);
    } else if (key === 'arrowleft' || key === 'p' || key === 'a') {
        e.preventDefault();
        navigateTo(currentIndex - 1);
    } else if (key === 's') {
        e.preventDefault();
        document.getElementById('saveBtn').click();
    } else if (key === '1' && selectedBoxIndex >= 0) {
        e.preventDefault();
        setSelectedBoxClass(1);
    } else if (key === '2' && selectedBoxIndex >= 0) {
        e.preventDefault();
        setSelectedBoxClass(2);
    } else if (key === '3' && selectedBoxIndex >= 0) {
        e.preventDefault();
        setSelectedBoxClass(3);
    } else if (key === '4' && selectedBoxIndex >= 0) {
        e.preventDefault();
        setSelectedBoxClass(4);
    } else if (key === '5' && selectedBoxIndex >= 0) {
        e.preventDefault();
        setSelectedBoxClass(0);
    } else if ((key === 'delete' || key === 'backspace') && selectedBoxIndex >= 0) {
        e.preventDefault();
        deleteBoxAtIndex(selectedBoxIndex);
        document.getElementById('hint').textContent = 'Deleted box. Save to persist.';
    }
}

window.addEventListener('keydown', handleShortcut, true);

if (dirSelect) {
    dirSelect.addEventListener('change', async () => {
        const selectedPath = dirSelect.value;
        if (!selectedPath || selectedPath === currentDirectoryPath) {
            return;
        }

        const priorPath = currentDirectoryPath;
        const priorHint = document.getElementById('hint').textContent;
        dirSelect.disabled = true;
        document.getElementById('hint').textContent = 'Switching working directory...';

        try {
            const result = await setWorkingDirectory(selectedPath);
            currentDirectoryPath = result.path;
            if (currentDirLabel) {
                currentDirLabel.textContent = `(current: ${result.name})`;
                currentDirLabel.title = result.path;
            }
            currentIndex = 0;
            images = [];
            currentImage = null;
            imageData = null;
            await loadImages();
            document.getElementById('hint').textContent = `Working directory switched to ${result.name}`;
        } catch (err) {
            console.error(err);
            if (priorPath) {
                dirSelect.value = priorPath;
            }
            document.getElementById('hint').textContent = `Directory switch failed: ${err.message || err}`;
        } finally {
            dirSelect.disabled = false;
            setTimeout(() => {
                if (document.getElementById('hint').textContent.includes('Working directory switched')) {
                    document.getElementById('hint').textContent = priorHint || 'Select a box to edit class/tags. Right-click deletes; middle-click sets vehicle. Save to persist.';
                }
            }, 2000);
        }
    });
}

async function initializeLookalikeTracker() {
    const autosaveInitial = localStorage.getItem('lookalike_autosave_on_navigate');
    autosaveOnNavigate = autosaveInitial === '1';
    const autosaveToggle = document.getElementById('autosaveToggle');
    if (autosaveToggle) {
        autosaveToggle.checked = autosaveOnNavigate;
    }

    applyZoom();

    await loadWorkingDirectories();
    await loadImages();
}

initializeLookalikeTracker().catch((err) => {
    console.error(err);
    document.getElementById('hint').textContent = `Initialization failed: ${err.message || err}`;
});
