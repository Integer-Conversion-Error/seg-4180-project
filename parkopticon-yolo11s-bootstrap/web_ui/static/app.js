const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let images = [];
let currentIndex = 0;
let currentImage = null;
let boxes = [];
let selectedBoxIndex = -1;
let isDragging = false;
let dragStart = { x: 0, y: 0 };
let scale = 1;
let offsetX = 0;
let offsetY = 0;
let imgElement = null;
let markInsertedMode = false;
let queueFilter = 'needs_review';
let autosaveOnNavigate = false;
let showExcluded = false;
let reviewSessionStartMs = Date.now();
let reviewedImageIds = new Set();
let currentDirectoryPath = null;
let hasUnsavedChanges = false;
let zoomLevel = 1;
let transitionInFlight = false;

const MIN_ZOOM = 0.25;
const MAX_ZOOM = 4;
const ZOOM_STEP = 0.25;

const dirSelect = document.getElementById('dirSelect');
const currentDirLabel = document.getElementById('currentDir');
const zoomSlider = document.getElementById('zoomSlider');
const zoomLevelLabel = document.getElementById('zoomLevel');
const saveStateBadge = document.getElementById('saveState');

const CLASS_NAMES = {
    0: 'vehicle',
    1: 'enforcement_vehicle',
    2: 'police_old',
    3: 'police_new',
    4: 'lookalike_negative'
};

const CLASS_COLORS = {
    0: '#4ade80',
    1: '#f59e0b',
    2: '#f97316',
    3: '#ef4444',
    4: '#a78bfa'
};

function expectedClassToId(expectedClass) {
    if (expectedClass === 'enforcement_vehicle') return 1;
    if (expectedClass === 'police_old') return 2;
    if (expectedClass === 'police_new') return 3;
    if (expectedClass === 'lookalike_negative') return 4;
    return 0;
}

function updateSaveStateBadge() {
    if (!saveStateBadge) {
        return;
    }
    if (hasUnsavedChanges) {
        saveStateBadge.textContent = 'Unsaved changes';
        saveStateBadge.classList.add('unsaved');
    } else {
        saveStateBadge.textContent = 'All changes saved';
        saveStateBadge.classList.remove('unsaved');
    }
}

function setUnsavedChanges(dirty) {
    hasUnsavedChanges = dirty;
    updateSaveStateBadge();
}

async function loadImages() {
    const resp = await fetch(
        `/api/labeler/images?queue=${encodeURIComponent(queueFilter)}&status=all&include_excluded=${showExcluded ? 'true' : 'false'}`
    );
    images = await resp.json();
    document.getElementById('progress').textContent = `${images.length} images in queue`;
    if (images.length > 0) {
        await loadImage(0);
    } else {
        currentIndex = 0;
        currentImage = null;
        boxes = [];
        selectedBoxIndex = -1;
        setUnsavedChanges(false);
        render();
    }
}

function resetReviewSession() {
    currentIndex = 0;
    currentImage = null;
    boxes = [];
    selectedBoxIndex = -1;
    setUnsavedChanges(false);
    reviewSessionStartMs = Date.now();
    reviewedImageIds = new Set();
    updateReviewSpeedMetric();
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

async function loadImage(index) {
    if (index < 0 || index >= images.length) return;
    
    currentIndex = index;
    currentImage = images[index];
    
    const resp = await fetch(`/api/labeler/image/${currentImage.image_id}`);
    const data = await resp.json();
    
    boxes = data.boxes || [];
    selectedBoxIndex = -1;
    setUnsavedChanges(false);
    
    document.getElementById('imageId').textContent = data.image_id.substring(0, 12) + '...';
    document.getElementById('imageStatus').textContent = data.is_synthetic ? 'Synthetic' : 'Original';
    document.getElementById('excludedStatus').textContent = data.is_excluded ? 'Yes' : 'No';
    document.getElementById('syntheticInfo').style.display = data.is_synthetic ? 'block' : 'none';
    document.getElementById('markInsertedBtn').style.display = data.is_synthetic ? 'inline-block' : 'none';
    
    if (!imgElement) {
        imgElement = new Image();
    }
    imgElement.onload = () => {
        canvas.width = imgElement.naturalWidth;
        canvas.height = imgElement.naturalHeight;
        applyZoom();
        render();
    };
    imgElement.src = data.image_path;
    
    updateBoxesList();
    updateProgress();
}

async function withTransitionLock(work) {
    if (transitionInFlight) {
        return false;
    }
    transitionInFlight = true;
    try {
        await work();
        return true;
    } finally {
        transitionInFlight = false;
    }
}

async function markCurrentImageReviewed() {
    if (!currentImage) {
        return;
    }
    if (currentImage.review_status !== 'done') {
        const resp = await fetch(`/api/labeler/review/${currentImage.image_id}?status=done`, {
            method: 'POST'
        });
        if (!resp.ok) {
            throw new Error(`Failed to mark reviewed (HTTP ${resp.status})`);
        }
        currentImage.review_status = 'done';
    }
    reviewedImageIds.add(currentImage.image_id);
    updateReviewSpeedMetric();
}

async function prepareCurrentImageForTransition() {
    if (!currentImage) {
        return true;
    }

    if (!hasUnsavedChanges) {
        if (autosaveOnNavigate) {
            await markCurrentImageReviewed();
        }
        return true;
    }

    if (autosaveOnNavigate) {
        await saveCurrentLabels(false);
        return true;
    }
    return window.confirm('You have unsaved box edits on this image. Navigate away and discard them?');
}

function buildYoloBoxes() {
    return boxes.map(box => {
        const x1 = box.x1 / canvas.width;
        const y1 = box.y1 / canvas.height;
        const x2 = box.x2 / canvas.width;
        const y2 = box.y2 / canvas.height;
        return { x1, y1, x2, y2, cls: box.cls };
    });
}

async function saveCurrentLabels(showMessage = true) {
    if (!currentImage) {
        return;
    }
    const yoloBoxes = buildYoloBoxes();
    const resp = await fetch(`/api/labeler/labels/${currentImage.image_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ boxes: yoloBoxes })
    });
    if (!resp.ok) {
        throw new Error(`Failed to save labels (HTTP ${resp.status})`);
    }
    setUnsavedChanges(false);
    currentImage.review_status = 'done';
    reviewedImageIds.add(currentImage.image_id);
    updateReviewSpeedMetric();
    if (showMessage) {
        document.getElementById('hint').textContent = 'Saved!';
        setTimeout(() => {
            document.getElementById('hint').textContent = 'Click and drag to create box';
        }, 1500);
    }
}

async function navigateTo(index) {
    if (index < 0 || index >= images.length) return;
    await withTransitionLock(async () => {
        try {
            const canLeave = await prepareCurrentImageForTransition();
            if (!canLeave) {
                return;
            }

            if (index < currentIndex) {
                const target = images[index];
                if (target && target.review_status === 'done') {
                    await fetch(`/api/labeler/review/${target.image_id}?status=todo`, { method: 'POST' });
                    target.review_status = 'todo';
                }
            }

            await loadImage(index);
        } catch (err) {
            console.error(err);
            const message = (err && err.message) ? err.message : String(err);
            document.getElementById('hint').textContent = `Navigation failed: ${message}`;
        }
    });
}

function clampZoom(value) {
    return Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, value));
}

function applyZoom() {
    if (!canvas.width || !canvas.height) {
        return;
    }
    canvas.style.width = `${Math.round(canvas.width * zoomLevel)}px`;
    canvas.style.height = `${Math.round(canvas.height * zoomLevel)}px`;
    if (zoomSlider) {
        zoomSlider.value = String(Math.round(zoomLevel * 100));
    }
    if (zoomLevelLabel) {
        zoomLevelLabel.textContent = `${Math.round(zoomLevel * 100)}%`;
    }
}

function setZoom(nextZoom) {
    zoomLevel = clampZoom(nextZoom);
    applyZoom();
}

function render() {
    if (!imgElement) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgElement, 0, 0);
    
    boxes.forEach((box, i) => {
        const isSelected = i === selectedBoxIndex;
        const color = CLASS_COLORS[box.cls] || '#fff';
        
        ctx.strokeStyle = color;
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
        
        if (isSelected) {
            ctx.fillStyle = color;
            ctx.fillRect(box.x1 - 2, box.y1 - 2, 4, 4);
            ctx.fillRect(box.x2 - 2, box.y1 - 2, 4, 4);
            ctx.fillRect(box.x1 - 2, box.y2 - 2, 4, 4);
            ctx.fillRect(box.x2 - 2, box.y2 - 2, 4, 4);
        }
        
        ctx.fillStyle = color;
        ctx.font = '14px sans-serif';
        ctx.fillText(CLASS_NAMES[box.cls] || 'unknown', box.x1, box.y1 - 5);
    });
}

function updateBoxesList() {
    const list = document.getElementById('boxesList');
    list.innerHTML = '';
    
    boxes.forEach((box, i) => {
        const item = document.createElement('div');
        item.className = `box-item class-${box.cls} ${i === selectedBoxIndex ? 'selected' : ''}`;
        item.innerHTML = `
            <div class="class-label">${CLASS_NAMES[box.cls]}</div>
            <div class="coords">${Math.round(box.x1)},${Math.round(box.y1)} - ${Math.round(box.x2)},${Math.round(box.y2)}</div>
        `;
        item.onclick = () => selectBox(i);
        list.appendChild(item);
    });
}

function selectBox(index) {
    if (markInsertedMode && currentImage && currentImage.expected_class) {
        const expectedCls = expectedClassToId(currentImage.expected_class);
        boxes[index].cls = expectedCls;
        setUnsavedChanges(true);
        markInsertedMode = false;
        document.getElementById('markInsertedBtn').textContent = 'Mark Inserted Vehicle';
        document.getElementById('hint').textContent = 'Click and drag to create box';
    } else {
        selectedBoxIndex = index;
    }
    render();
    updateBoxesList();
}

function updateProgress() {
    document.getElementById('progress').textContent = `${currentIndex + 1} / ${images.length}`;
}

function updateReviewSpeedMetric() {
    const elapsedMin = Math.max((Date.now() - reviewSessionStartMs) / 60000, 1 / 60);
    const reviewed = reviewedImageIds.size;
    const speed = reviewed / elapsedMin;
    const speedEl = document.getElementById('reviewSpeed');
    if (speedEl) {
        speedEl.textContent = `Reviewed: ${reviewed} | Speed: ${speed.toFixed(1)}/min`;
    }
}

function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const displayWidth = canvas.clientWidth || rect.width;
    const displayHeight = canvas.clientHeight || rect.height;
    const scaleX = canvas.width / displayWidth;
    const scaleY = canvas.height / displayHeight;
    return {
        x: (e.clientX - rect.left - canvas.clientLeft) * scaleX,
        y: (e.clientY - rect.top - canvas.clientTop) * scaleY
    };
}

function findBoxIndexAtCoords(coords) {
    for (let i = boxes.length - 1; i >= 0; i--) {
        const box = boxes[i];
        if (coords.x >= box.x1 && coords.x <= box.x2 && coords.y >= box.y1 && coords.y <= box.y2) {
            return i;
        }
    }
    return -1;
}

canvas.addEventListener('mousedown', (e) => {
    if (e.button !== 0) {
        if (e.button === 1) {
            e.preventDefault();
        }
        return;
    }

    const coords = getCanvasCoords(e);
    dragStart = coords;
    isDragging = true;

    const hitIndex = findBoxIndexAtCoords(coords);
    if (hitIndex >= 0) {
        selectedBoxIndex = hitIndex;
        render();
        updateBoxesList();
        return;
    }

    selectedBoxIndex = -1;
    render();
    updateBoxesList();
});

canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    const coords = getCanvasCoords(e);
    const hitIndex = findBoxIndexAtCoords(coords);
    if (hitIndex < 0) {
        return;
    }

    boxes.splice(hitIndex, 1);
    if (selectedBoxIndex === hitIndex) {
        selectedBoxIndex = -1;
    } else if (selectedBoxIndex > hitIndex) {
        selectedBoxIndex -= 1;
    }
    setUnsavedChanges(true);
    render();
    updateBoxesList();
    const hint = document.getElementById('hint');
    if (hint) {
        hint.textContent = 'Deleted box via right-click';
    }
});

canvas.addEventListener('auxclick', (e) => {
    if (e.button !== 1) {
        return;
    }
    e.preventDefault();

    const coords = getCanvasCoords(e);
    const hitIndex = findBoxIndexAtCoords(coords);
    if (hitIndex < 0) {
        return;
    }

    boxes[hitIndex].cls = 0;
    selectedBoxIndex = hitIndex;
    setUnsavedChanges(true);
    render();
    updateBoxesList();

    const hint = document.getElementById('hint');
    if (hint) {
        hint.textContent = 'Set box class to vehicle via middle-click';
    }
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    
    const coords = getCanvasCoords(e);
    
    render();
    
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(
        dragStart.x,
        dragStart.y,
        coords.x - dragStart.x,
        coords.y - dragStart.y
    );
    ctx.setLineDash([]);
});

canvas.addEventListener('mouseup', (e) => {
    if (!isDragging) return;
    isDragging = false;
    
    const coords = getCanvasCoords(e);
    
    let x1 = Math.min(dragStart.x, coords.x);
    let y1 = Math.min(dragStart.y, coords.y);
    let x2 = Math.max(dragStart.x, coords.x);
    let y2 = Math.max(dragStart.y, coords.y);
    
    if (x2 - x1 > 10 && y2 - y1 > 10) {
        if (selectedBoxIndex >= 0) {
            boxes[selectedBoxIndex] = { x1, y1, x2, y2, cls: boxes[selectedBoxIndex].cls };
        } else {
            boxes.push({ x1, y1, x2, y2, cls: 0 });
            selectedBoxIndex = boxes.length - 1;
        }
        setUnsavedChanges(true);
        render();
        updateBoxesList();
    }
});

document.getElementById('prevBtn').onclick = () => navigateTo(currentIndex - 1);
document.getElementById('nextBtn').onclick = () => navigateTo(currentIndex + 1);

document.getElementById('queueFilter').onchange = async (e) => {
    await withTransitionLock(async () => {
        const canLeave = await prepareCurrentImageForTransition();
        if (!canLeave) {
            e.target.value = queueFilter;
            return;
        }
        queueFilter = e.target.value;
        currentIndex = 0;
        reviewSessionStartMs = Date.now();
        reviewedImageIds = new Set();
        updateReviewSpeedMetric();
        await loadImages();
    });
};

document.getElementById('showExcludedToggle').onchange = async (e) => {
    await withTransitionLock(async () => {
        const canLeave = await prepareCurrentImageForTransition();
        if (!canLeave) {
            e.target.checked = showExcluded;
            return;
        }
        showExcluded = !!e.target.checked;
        currentIndex = 0;
        reviewSessionStartMs = Date.now();
        reviewedImageIds = new Set();
        updateReviewSpeedMetric();
        await loadImages();
    });
};

if (dirSelect) {
    dirSelect.addEventListener('change', async () => {
        await withTransitionLock(async () => {
            const selectedPath = dirSelect.value;
            if (!selectedPath || selectedPath === currentDirectoryPath) {
                return;
            }

            const canLeave = await prepareCurrentImageForTransition();
            if (!canLeave) {
                if (currentDirectoryPath) {
                    dirSelect.value = currentDirectoryPath;
                }
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
                resetReviewSession();
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
                        document.getElementById('hint').textContent = priorHint || 'Click and drag to create box';
                    }
                }, 2000);
            }
        });
    });
}

async function jumpToImageId(rawId) {
    const imageId = (rawId || '').trim();
    if (!imageId) return;

    const exactIndex = images.findIndex(img => img.image_id === imageId);
    if (exactIndex >= 0) {
        await navigateTo(exactIndex);
        return;
    }

    const prefixIndex = images.findIndex(img => img.image_id.startsWith(imageId));
    if (prefixIndex >= 0) {
        await navigateTo(prefixIndex);
        return;
    }

    const resp = await fetch(`/api/labeler/image/${imageId}`);
    if (!resp.ok) {
        document.getElementById('hint').textContent = `Image not found: ${imageId}`;
        return;
    }

    images.push({ image_id: imageId, review_status: 'todo' });
    await navigateTo(images.length - 1);
    document.getElementById('hint').textContent = `Loaded image ${imageId} (outside current queue)`;
}

document.getElementById('imageSearchBtn').onclick = async () => {
    await jumpToImageId(document.getElementById('imageSearchInput').value);
};

document.getElementById('imageSearchInput').addEventListener('keydown', async (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        await jumpToImageId(e.target.value);
    }
});

document.getElementById('addBoxBtn').onclick = () => {
    boxes.push({ x1: 100, y1: 100, x2: 300, y2: 200, cls: 0 });
    selectedBoxIndex = boxes.length - 1;
    setUnsavedChanges(true);
    render();
    updateBoxesList();
};

document.getElementById('saveBtn').onclick = async () => {
    await saveCurrentLabels(true);
};

document.getElementById('autosaveToggle').onchange = (e) => {
    autosaveOnNavigate = !!e.target.checked;
    localStorage.setItem('labeler_autosave_on_navigate', autosaveOnNavigate ? '1' : '0');
};

document.getElementById('markInsertedBtn').onclick = () => {
    markInsertedMode = !markInsertedMode;
    document.getElementById('markInsertedBtn').textContent = markInsertedMode ? 
        'Click box to mark →' : 'Mark Inserted Vehicle';
    document.getElementById('hint').textContent = markInsertedMode ?
        'Click on the inserted vehicle box' : 'Click and drag to create box';
};

function handleShortcut(e) {
    const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : '';
    if (tag === 'input' || tag === 'textarea' || e.target?.isContentEditable) {
        return;
    }

    const key = (e.key || '').toLowerCase();

    if (key === 'arrowright' || key === 'd' || key === 'n') {
        e.preventDefault();
        navigateTo(currentIndex + 1);
    } else if (key === 'arrowleft' || key === 'a' || key === 'p') {
        e.preventDefault();
        navigateTo(currentIndex - 1);
    } else if (key === 's') {
        e.preventDefault();
        document.getElementById('saveBtn').click();
    } else if (key === '5' && selectedBoxIndex >= 0) {
        boxes[selectedBoxIndex].cls = 0;
        setUnsavedChanges(true);
        render();
        updateBoxesList();
    } else if (key === '1' && selectedBoxIndex >= 0) {
        boxes[selectedBoxIndex].cls = 1;
        setUnsavedChanges(true);
        render();
        updateBoxesList();
    } else if (key === '2' && selectedBoxIndex >= 0) {
        boxes[selectedBoxIndex].cls = 2;
        setUnsavedChanges(true);
        render();
        updateBoxesList();
    } else if (key === '3' && selectedBoxIndex >= 0) {
        boxes[selectedBoxIndex].cls = 3;
        setUnsavedChanges(true);
        render();
        updateBoxesList();
    } else if (key === '4' && selectedBoxIndex >= 0) {
        boxes[selectedBoxIndex].cls = 4;
        setUnsavedChanges(true);
        render();
        updateBoxesList();
    } else if (key === 'delete' && selectedBoxIndex >= 0) {
        boxes.splice(selectedBoxIndex, 1);
        selectedBoxIndex = -1;
        setUnsavedChanges(true);
        render();
        updateBoxesList();
    }
}

window.addEventListener('keydown', handleShortcut, true);

const autosaveInitial = localStorage.getItem('labeler_autosave_on_navigate');
autosaveOnNavigate = autosaveInitial !== '0';
const autosaveToggle = document.getElementById('autosaveToggle');
if (autosaveToggle) {
    autosaveToggle.checked = autosaveOnNavigate;
}

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

applyZoom();
updateSaveStateBadge();

window.addEventListener('beforeunload', (e) => {
    if (!hasUnsavedChanges) {
        return;
    }
    e.preventDefault();
    e.returnValue = '';
});

const showExcludedToggle = document.getElementById('showExcludedToggle');
if (showExcludedToggle) {
    showExcludedToggle.checked = showExcluded;
}

updateReviewSpeedMetric();

async function initializeLabeler() {
    await loadWorkingDirectories();
    await loadImages();

    const params = new URLSearchParams(window.location.search || '');
    const initialImageId = (params.get('image_id') || '').trim();
    if (initialImageId) {
        await jumpToImageId(initialImageId);
    }
}

initializeLabeler();
