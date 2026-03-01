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

const CLASS_NAMES = { 0: 'vehicle', 1: 'enforcement_vehicle' };
const CLASS_COLORS = { 0: '#4ade80', 1: '#f59e0b' };

async function loadImages() {
    const resp = await fetch('/api/images');
    images = await resp.json();
    document.getElementById('progress').textContent = `${images.length} images to review`;
    if (images.length > 0) {
        await loadImage(0);
    }
}

async function loadImage(index) {
    if (index < 0 || index >= images.length) return;
    
    currentIndex = index;
    currentImage = images[index];
    
    const resp = await fetch(`/api/image/${currentImage.image_id}`);
    const data = await resp.json();
    
    boxes = data.boxes || [];
    selectedBoxIndex = -1;
    
    document.getElementById('imageId').textContent = data.image_id.substring(0, 12) + '...';
    document.getElementById('imageStatus').textContent = data.is_synthetic ? 'Synthetic' : 'Original';
    document.getElementById('syntheticInfo').style.display = data.is_synthetic ? 'block' : 'none';
    document.getElementById('markInsertedBtn').style.display = data.is_synthetic ? 'inline-block' : 'none';
    
    if (imgElement) {
        imgElement.onload = () => {
            render();
        };
        imgElement.src = data.image_path;
    } else {
        imgElement = new Image();
        imgElement.onload = () => {
            canvas.width = imgElement.width;
            canvas.height = imgElement.height;
            render();
        };
        imgElement.src = data.image_path;
    }
    
    updateBoxesList();
    updateProgress();
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
        const expectedCls = currentImage.expected_class === 'enforcement_vehicle' ? 1 : 0;
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

document.getElementById('prevBtn').onclick = () => loadImage(currentIndex - 1);
document.getElementById('nextBtn').onclick = () => loadImage(currentIndex + 1);

document.getElementById('addBoxBtn').onclick = () => {
    boxes.push({ x1: 100, y1: 100, x2: 300, y2: 200, cls: 0 });
    selectedBoxIndex = boxes.length - 1;
    render();
    updateBoxesList();
};

document.getElementById('saveBtn').onclick = async () => {
    const yoloBoxes = boxes.map(box => {
        const x_center = (box.x1 + box.x2) / 2 / canvas.width;
        const y_center = (box.y1 + box.y2) / 2 / canvas.height;
        const width = (box.x2 - box.x1) / canvas.width;
        const height = (box.y2 - box.y1) / canvas.height;
        return { x1: x_center, y1: y_center, x2: x_center + width, y2: y_center + height, cls: box.cls };
    });
    
    await fetch(`/api/labels/${currentImage.image_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ boxes: yoloBoxes })
    });
    
    document.getElementById('hint').textContent = 'Saved!';
    setTimeout(() => {
        document.getElementById('hint').textContent = 'Click and drag to create box';
    }, 1500);
};

document.getElementById('markInsertedBtn').onclick = () => {
    markInsertedMode = !markInsertedMode;
    document.getElementById('markInsertedBtn').textContent = markInsertedMode ? 
        'Click box to mark →' : 'Mark Inserted Vehicle';
    document.getElementById('hint').textContent = markInsertedMode ?
        'Click on the inserted vehicle box' : 'Click and drag to create box';
};

document.addEventListener('keydown', (e) => {
    if (e.key === 'n' || e.key === 'N') {
        loadImage(currentIndex + 1);
    } else if (e.key === 'p' || e.key === 'P') {
        loadImage(currentIndex - 1);
    } else if (e.key === 's' || e.key === 'S') {
        document.getElementById('saveBtn').click();
    } else if (e.key === '0' && selectedBoxIndex >= 0) {
        boxes[selectedBoxIndex].cls = 0;
        render();
        updateBoxesList();
    } else if (e.key === '1' && selectedBoxIndex >= 0) {
        boxes[selectedBoxIndex].cls = 1;
        render();
        updateBoxesList();
    } else if (e.key === 'Delete' && selectedBoxIndex >= 0) {
        boxes.splice(selectedBoxIndex, 1);
        selectedBoxIndex = -1;
        render();
        updateBoxesList();
    }
});

loadImages();
