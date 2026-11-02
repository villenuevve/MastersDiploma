(() => {
    "use strict";

    const image = document.getElementById("resultImage");

    if (!image) {
        console.error("Inspector: #resultImage not found");
        return;
    }

    const CLASS_NAMES = {
        0: "road",
        1: "sidewalk",
        2: "building",
        3: "wall",
        4: "fence",
        5: "pole",
        6: "traffic light",
        7: "traffic sign",
        8: "vegetation",
        9: "terrain",
        10: "sky",
        11: "person",
        12: "rider",
        13: "car",
        14: "truck",
        15: "bus",
        16: "train",
        17: "motorcycle",
        18: "bicycle"
    };

    let overlay = null;
    let ctx = null;

    let maskPixels = null;
    let maskWidth = 0;
    let maskHeight = 0;

    let componentMap = null;
    let components = new Map();
    let nextComponentId = 1;

    let classStats = [];
    let activeComponent = null;

    // Tooltip
    const tooltip = document.createElement("div");
    tooltip.id = "segmentationInspectorTooltip";

    Object.assign(tooltip.style, {
        position: "fixed",
        display: "none",
        zIndex: "99999",
        pointerEvents: "none",
        padding: "10px 14px",
        borderRadius: "9px",
        background: "rgba(20, 24, 32, 0.96)",
        color: "#ffffff",
        font: "14px/1.5 -apple-system, BlinkMacSystemFont, sans-serif",
        boxShadow: "0 4px 18px rgba(0,0,0,.25)",
        whiteSpace: "nowrap"
    });

    document.body.appendChild(tooltip);

    // Create highlight canvas
    function createOverlay() {
        const parent = image.parentElement;
        if (!parent) return;

        if (!overlay) {
            if (getComputedStyle(parent).position === "static") {
                parent.style.position = "relative";
            }

            overlay = document.createElement("canvas");
            overlay.id = "segmentationHighlightCanvas";

            Object.assign(overlay.style, {
                position: "absolute",
                pointerEvents: "none",
                display: "none",
                zIndex: "5"
            });

            parent.appendChild(overlay);

            ctx = overlay.getContext("2d", {
                willReadFrequently: true
            });
        }

        positionOverlay();
    }

    // Align canvas with the displayed image
    function positionOverlay() {
        if (!overlay || !image.parentElement) return;

        const parent = image.parentElement;

        const imageRect = image.getBoundingClientRect();
        const parentRect = parent.getBoundingClientRect();

        overlay.style.left =
            `${imageRect.left - parentRect.left + parent.scrollLeft}px`;

        overlay.style.top =
            `${imageRect.top - parentRect.top + parent.scrollTop}px`;

        overlay.style.width = `${imageRect.width}px`;
        overlay.style.height = `${imageRect.height}px`;
    }

    // Clear selected region
    function clearHighlight() {
        activeComponent = null;

        if (ctx && overlay) {
            ctx.clearRect(0, 0, overlay.width, overlay.height);
            overlay.style.display = "none";
        }

        tooltip.style.display = "none";
    }

    // Get class statistics from API response
    function getStat(classId) {
        return classStats.find(item =>
            Number(item.id) === Number(classId)
        );
    }

    function getClassName(classId) {
        const stat = getStat(classId);

        return stat?.name ??
            CLASS_NAMES[classId] ??
            `Class ${classId}`;
    }

    function getPercentage(classId) {
        const stat = getStat(classId);

        if (!stat || stat.percentage == null) {
            return null;
        }

        const value = Number(stat.percentage);

        return Number.isFinite(value) ? value : null;
    }

    // Decode class ID mask
    function prepareMask(maskImage) {
        const scale = Math.min(
            1,
            640 / Math.max(
                maskImage.naturalWidth,
                maskImage.naturalHeight
            )
        );

        maskWidth = Math.max(
            1,
            Math.round(maskImage.naturalWidth * scale)
        );

        maskHeight = Math.max(
            1,
            Math.round(maskImage.naturalHeight * scale)
        );

        const temp = document.createElement("canvas");

        temp.width = maskWidth;
        temp.height = maskHeight;

        const tempCtx = temp.getContext("2d", {
            willReadFrequently: true
        });

        tempCtx.imageSmoothingEnabled = false;

        tempCtx.drawImage(
            maskImage,
            0,
            0,
            maskWidth,
            maskHeight
        );

        const pixels = tempCtx.getImageData(
            0,
            0,
            maskWidth,
            maskHeight
        ).data;

        maskPixels = new Uint8Array(
            maskWidth * maskHeight
        );

        // Class IDs are stored in the red channel.
        for (let i = 0; i < maskPixels.length; i++) {
            maskPixels[i] = pixels[i * 4];
        }

        componentMap = new Int32Array(maskPixels.length);

        components.clear();
        nextComponentId = 1;

        createOverlay();

        overlay.width = maskWidth;
        overlay.height = maskHeight;

        clearHighlight();

        console.log(
            `Inspector: mask loaded ${maskWidth} × ${maskHeight}`
        );
    }

    function loadMask(base64) {
        if (!base64) {
            console.warn(
                "Inspector: class_ids_base64 is missing"
            );

            maskPixels = null;
            clearHighlight();

            return;
        }

        const maskImage = new Image();

        maskImage.onload = () => {
            prepareMask(maskImage);
        };

        maskImage.onerror = () => {
            console.error(
                "Inspector: failed to load class ID mask"
            );

            maskPixels = null;
            clearHighlight();
        };

        maskImage.src = base64.startsWith("data:")
            ? base64
            : `data:image/png;base64,${base64}`;
    }

    // Find connected region with the same class ID
    function findComponent(seed) {
        const classId = maskPixels[seed];

        if (componentMap[seed] !== 0) {
            const id = componentMap[seed];

            return {
                id,
                classId,
                pixels: components.get(id)
            };
        }

        const componentId = nextComponentId++;

        const queue = new Int32Array(maskPixels.length);
        const pixels = [];

        let head = 0;
        let tail = 0;

        queue[tail++] = seed;
        componentMap[seed] = componentId;

        while (head < tail) {
            const current = queue[head++];

            pixels.push(current);

            const x = current % maskWidth;
            const y = Math.floor(current / maskWidth);

            const neighbors = [
                x > 0 ? current - 1 : -1,
                x < maskWidth - 1 ? current + 1 : -1,
                y > 0 ? current - maskWidth : -1,
                y < maskHeight - 1 ? current + maskWidth : -1
            ];

            for (const neighbor of neighbors) {
                if (
                    neighbor >= 0 &&
                    componentMap[neighbor] === 0 &&
                    maskPixels[neighbor] === classId
                ) {
                    componentMap[neighbor] = componentId;
                    queue[tail++] = neighbor;
                }
            }
        }

        components.set(componentId, pixels);

        return {
            id: componentId,
            classId,
            pixels
        };
    }

    // Draw selected region
    function drawComponent(component) {
        const output = ctx.createImageData(
            maskWidth,
            maskHeight
        );

        const data = output.data;

        // Darken the rest of the image.
        for (let i = 0; i < maskPixels.length; i++) {
            const p = i * 4;

            data[p] = 0;
            data[p + 1] = 0;
            data[p + 2] = 0;
            data[p + 3] = 112;
        }

        // Highlight selected connected region.
        for (const index of component.pixels) {
            const x = index % maskWidth;
            const y = Math.floor(index / maskWidth);

            const p = index * 4;

            let boundary =
                x === 0 ||
                x === maskWidth - 1 ||
                y === 0 ||
                y === maskHeight - 1;

            if (!boundary) {
                boundary =
                    componentMap[index - 1] !== component.id ||
                    componentMap[index + 1] !== component.id ||
                    componentMap[index - maskWidth] !== component.id ||
                    componentMap[index + maskWidth] !== component.id;
            }

            data[p] = 30;
            data[p + 1] = 200;
            data[p + 2] = 255;
            data[p + 3] = boundary ? 225 : 65;
        }

        ctx.putImageData(output, 0, 0);

        positionOverlay();

        overlay.style.display = "block";
    }

    // Show class name and percentage
    function showTooltip(event, classId) {
        const name = getClassName(classId);
        const percentage = getPercentage(classId);

        tooltip.replaceChildren();

        const title = document.createElement("div");

        title.textContent = name;
        title.style.fontWeight = "600";

        const detail = document.createElement("div");

        detail.textContent = percentage === null
            ? "Percentage unavailable"
            : `${percentage.toFixed(2)}% of image`;

        detail.style.opacity = "0.85";

        tooltip.append(title, detail);

        tooltip.style.display = "block";

        const margin = 14;

        const left = Math.min(
            event.clientX + margin,
            window.innerWidth - tooltip.offsetWidth - 8
        );

        const top = Math.min(
            event.clientY + margin,
            window.innerHeight - tooltip.offsetHeight - 8
        );

        tooltip.style.left = `${Math.max(8, left)}px`;
        tooltip.style.top = `${Math.max(8, top)}px`;
    }

    // Map cursor position to mask coordinates
    function getImageCoordinates(event) {
        const rect = image.getBoundingClientRect();

        if (!rect.width || !rect.height) return null;

        const imageRatio =
            image.naturalWidth / image.naturalHeight;

        const containerRatio =
            rect.width / rect.height;

        let displayedWidth;
        let displayedHeight;
        let offsetX = 0;
        let offsetY = 0;

        if (imageRatio > containerRatio) {
            displayedWidth = rect.width;
            displayedHeight = rect.width / imageRatio;

            offsetY = (rect.height - displayedHeight) / 2;
        } else {
            displayedHeight = rect.height;
            displayedWidth = rect.height * imageRatio;

            offsetX = (rect.width - displayedWidth) / 2;
        }

        const localX =
            event.clientX - rect.left - offsetX;

        const localY =
            event.clientY - rect.top - offsetY;

        if (
            localX < 0 ||
            localY < 0 ||
            localX >= displayedWidth ||
            localY >= displayedHeight
        ) {
            return null;
        }

        return {
            x: Math.floor(
                localX / displayedWidth * maskWidth
            ),
            y: Math.floor(
                localY / displayedHeight * maskHeight
            )
        };
    }

    // Mouse interaction
    image.addEventListener("mousemove", event => {
        if (
            !maskPixels ||
            !image.naturalWidth ||
            !image.naturalHeight
        ) {
            return;
        }

        const point = getImageCoordinates(event);

        if (!point) {
            clearHighlight();
            return;
        }

        const seed =
            point.y * maskWidth + point.x;

        if (
            seed < 0 ||
            seed >= maskPixels.length
        ) {
            return;
        }

        const component = findComponent(seed);

        if (
            !activeComponent ||
            activeComponent.id !== component.id
        ) {
            activeComponent = component;
            drawComponent(component);
        }

        showTooltip(event, maskPixels[seed]);
    });

    image.addEventListener("mouseleave", clearHighlight);

    window.addEventListener("resize", () => {
        positionOverlay();
        clearHighlight();
    });

    // Receive new segmentation result
    window.addEventListener("segmentation-result", event => {
        const data = event.detail || {};

        classStats = Array.isArray(data.detected_classes)
            ? data.detected_classes
            : [];

        clearHighlight();

        loadMask(data.class_ids_base64);

        console.log(
            "Inspector: class statistics loaded",
            classStats.length
        );
    });

    console.log("Segmentation Inspector initialized");
})();