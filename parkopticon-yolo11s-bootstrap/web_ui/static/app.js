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

async function loadImages() {
    const resp = await fetch(
        `/api/labeler/images?queue=${encodeURIComponent(queueFilter)}&include_excluded=${showExcluded ? 'true' : 'false'}`
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
        render();
    }
}

async function loadImage(index) {
    if (index < 0 || index >= images.length) return;
    
    currentIndex = index;
    currentImage = images[index];
    
    const resp = await fetch(`/api/labeler/image/${currentImage.image_id}`);
    const data = await resp.json();
    
    boxes = data.boxes || [];
    selectedBoxIndex = -1;
    
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
        render();
    };
    imgElement.src = data.image_path;
    
    updateBoxesList();
    updateProgress();
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
    await fetch(`/api/labeler/labels/${currentImage.image_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ boxes: yoloBoxes })
    });
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

    if (index < currentIndex) {
        const target = images[index];
        if (target && target.review_status === 'done') {
            await fetch(`/api/labeler/review/${target.image_id}?status=todo`, { method: 'POST' });
            target.review_status = 'todo';
        }
    }

    if (autosaveOnNavigate && currentImage) {
        await saveCurrentLabels(false);
    }
    await loadImage(index);
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
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

canvas.addEventListener('mousedown', (e) => {
    const coords = getCanvasCoords(e);
    dragStart = coords;
    isDragging = true;
    
    for (let i = 0; i < boxes.length; i++) {
        const box = boxes[i];
        if (coords.x >= box.x1 && coords.x <= box.x2 && coords.y >= box.y1 && coords.y <= box.y2) {
            selectedBoxIndex = i;
            render();
            updateBoxesList();
            return;
        }
    }
    
    selectedBoxIndex = -1;
    render();
    updateBoxesList();
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
        render();
        updateBoxesList();
    }
});

document.getElementById('prevBtn').onclick = () => navigateTo(currentIndex - 1);
document.getElementById('nextBtn').onclick = () => navigateTo(currentIndex + 1);

document.getElementById('queueFilter').onchange = async (e) => {
    queueFilter = e.target.value;
    currentIndex = 0;
    reviewSessionStartMs = Date.now();
    reviewedImageIds = new Set();
    updateReviewSpeedMetric();
    await loadImages();
};

document.getElementById('showExcludedToggle').onchange = async (e) => {
    showExcluded = !!e.target.checked;
    currentIndex = 0;
    reviewSessionStartMs = Date.now();
    reviewedImageIds = new Set();
    updateReviewSpeedMetric();
    await loadImages();
};

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
    } else if (key === '0' && selectedBoxIndex >= 0) {
        boxes[selectedBoxIndex].cls = 0;
        render();
        updateBoxesList();
    } else if (key === '1' && selectedBoxIndex >= 0) {
        boxes[selectedBoxIndex].cls = 1;
        render();
        updateBoxesList();
    } else if (key === '2' && selectedBoxIndex >= 0) {
        boxes[selectedBoxIndex].cls = 2;
        render();
        updateBoxesList();
    } else if (key === '3' && selectedBoxIndex >= 0) {
        boxes[selectedBoxIndex].cls = 3;
        render();
        updateBoxesList();
    } else if (key === '4' && selectedBoxIndex >= 0) {
        boxes[selectedBoxIndex].cls = 4;
        render();
        updateBoxesList();
    } else if (key === 'delete' && selectedBoxIndex >= 0) {
        boxes.splice(selectedBoxIndex, 1);
        selectedBoxIndex = -1;
        render();
        updateBoxesList();
    }
}

window.addEventListener('keydown', handleShortcut, true);

const autosaveInitial = localStorage.getItem('labeler_autosave_on_navigate');
autosaveOnNavigate = autosaveInitial === '1';
const autosaveToggle = document.getElementById('autosaveToggle');
if (autosaveToggle) {
    autosaveToggle.checked = autosaveOnNavigate;
}

const showExcludedToggle = document.getElementById('showExcludedToggle');
if (showExcludedToggle) {
    showExcludedToggle.checked = showExcluded;
}

updateReviewSpeedMetric();

loadImages();
