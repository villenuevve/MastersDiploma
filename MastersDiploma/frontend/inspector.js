(() => {
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

    let overlay;
    let ctx;
    let maskPixels;
    let maskWidth = 0;
    let maskHeight = 0;
    let componentMap;
    let nextComponentId = 1;
    let components = new Map();
    let classStats = [];
    let activeComponent = null;
    let lastCell = -1;
    let currentMaskUrl = null;

    const tooltip = document.createElement("div");
    tooltip.id = "segmentationInspectorTooltip";
    Object.assign(tooltip.style, {
        position: "fixed",
        display: "none",
        zIndex: "99999",
        pointerEvents: "none",
        padding: "9px 12px",
        borderRadius: "8px",
        background: "rgba(20, 24, 32, 0.96)",
        color: "#fff",
        font: "13px/1.5 -apple-system, BlinkMacSystemFont, sans-serif",
        boxShadow: "0 4px 18px rgba(0,0,0,.25)",
        whiteSpace: "nowrap"
    });
    document.body.appendChild(tooltip);

    function createOverlay() {
        if (!image.parentElement) return;

        if (!overlay) {
            const parent = image.parentElement;
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
            ctx = overlay.getContext("2d", { willReadFrequently: true });
        }

        const imageRect = image.getBoundingClientRect();
        const parentRect = image.parentElement.getBoundingClientRect();

        overlay.style.left =
            (imageRect.left - parentRect.left + image.parentElement.scrollLeft) + "px";
        overlay.style.top =
            (imageRect.top - parentRect.top + image.parentElement.scrollTop) + "px";
        overlay.style.width = imageRect.width + "px";
        overlay.style.height = imageRect.height + "px";
    }

    function clearHighlight() {
        activeComponent = null;
        lastCell = -1;
        if (ctx && overlay) {
            ctx.clearRect(0, 0, overlay.width, overlay.height);
            overlay.style.display = "none";
        }
        tooltip.style.display = "none";
    }

    function getStat(classId) {
        return classStats.find(item =>
            Number(item.class_id ?? item.id) === classId
        );
    }

    function getClassName(classId) {
        const stat = getStat(classId);
        return stat?.class_name ?? stat?.name ?? CLASS_NAMES[classId] ?? `Class ${classId}`;
    }

    function getPercentage(classId) {
        const stat = getStat(classId);
        const value = stat?.percentage ?? stat?.percent;
        if (value === undefined || value === null) return null;
        return Number(value);
    }

    function prepareMask(maskImage) {
        const scale = Math.min(1, 640 / Math.max(
            maskImage.naturalWidth,
            maskImage.naturalHeight
        ));

        maskWidth = Math.max(1, Math.round(maskImage.naturalWidth * scale));
        maskHeight = Math.max(1, Math.round(maskImage.naturalHeight * scale));

        const temp = document.createElement("canvas");
        temp.width = maskWidth;
        temp.height = maskHeight;

        const tempCtx = temp.getContext("2d", { willReadFrequently: true });
        tempCtx.imageSmoothingEnabled = false;
        tempCtx.drawImage(maskImage, 0, 0, maskWidth, maskHeight);

        const pixels = tempCtx.getImageData(0, 0, maskWidth, maskHeight).data;
        maskPixels = new Uint8Array(maskWidth * maskHeight);

        for (let i = 0; i < maskPixels.length; i++) {
            maskPixels[i] = pixels[i * 4];
        }

        componentMap = new Int32Array(maskPixels.length);
        components.clear();
        nextComponentId = 1;

        overlay.width = maskWidth;
        overlay.height = maskHeight;

        createOverlay();
        clearHighlight();
    }

    function findComponent(seed) {
        const classId = maskPixels[seed];

        if (componentMap[seed] !== 0) {
            return {
                id: componentMap[seed],
                classId,
                pixels: components.get(componentMap[seed])
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
        return { id: componentId, classId, pixels };
    }

    function drawComponent(component) {
        const output = ctx.createImageData(maskWidth, maskHeight);
        const data = output.data;

        // Dim all pixels outside the selected connected region.
        for (let i = 0; i < maskPixels.length; i++) {
            const p = i * 4;
            data[p] = 0;
            data[p + 1] = 0;
            data[p + 2] = 0;
            data[p + 3] = 112;
        }

        // Highlight the selected region and emphasize its boundary.
        for (const index of component.pixels) {
            const x = index % maskWidth;
            const y = Math.floor(index / maskWidth);
            const p = index * 4;

            let boundary = false;

            if (x === 0 || x === maskWidth - 1 ||
                y === 0 || y === maskHeight - 1) {
                boundary = true;
            } else {
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
        createOverlay();
        overlay.style.display = "block";
    }

    function showTooltip(event, classId) {
        const name = getClassName(classId);
        const percentage = getPercentage(classId);

        tooltip.innerHTML = "";

        const title = document.createElement("div");
        title.textContent = name;
        title.style.fontWeight = "600";
        tooltip.appendChild(title);

        const detail = document.createElement("div");
        detail.textContent = percentage === null
            ? "Percentage unavailable"
            : `${percentage.toFixed(2)}% of image`;
        detail.style.opacity = "0.85";
        tooltip.appendChild(detail);

        tooltip.style.display = "block";

        const margin = 14;
        const width = tooltip.offsetWidth;
        const height = tooltip.offsetHeight;

        tooltip.style.left = Math.min(
            event.clientX + margin,
            window.innerWidth - width - 8
        ) + "px";

        tooltip.style.top = Math.min(
            event.clientY + margin,
            window.innerHeight - height - 8
        ) + "px";
    }

    image.addEventListener("mousemove", event => {
        if (!maskPixels || !image.naturalWidth || !image.naturalHeight) return;

        const rect = image.getBoundingClientRect();
        if (!rect.width || !rect.height) return;

        const x = Math.max(0, Math.min(
            maskWidth - 1,
            Math.floor((event.clientX - rect.left) / rect.width * maskWidth)
        ));
        const y = Math.max(0, Math.min(
            maskHeight - 1,
            Math.floor((event.clientY - rect.top) / rect.height * maskHeight)
        ));

        const seed = y * maskWidth + x;
        if (seed < 0 || seed >= maskPixels.length) return;

        const componentId = componentMap[seed];

        if (seed !== lastCell) {
            const component = findComponent(seed);

            if (!activeComponent || activeComponent.id !== component.id) {
                activeComponent = component;
                drawComponent(component);
            }

            lastCell = seed;
        }

        showTooltip(event, maskPixels[seed]);
    });

    image.addEventListener("mouseleave", clearHighlight);
    window.addEventListener("resize", () => {
        if (overlay && maskPixels) createOverlay();
    });

    window.addEventListener("segmentation-result", event => {
        const data = event.detail || {};
        classStats = Array.isArray(data.detected_classes)
            ? data.detected_classes
            : [];

        const encoded = data.class_ids_base64;
        if (!encoded) {
            console.error("Inspector: class_ids_base64 missing from API response");
            clearHighlight();
            return;
        }

        const cleanBase64 = encoded.replace(/^data:image\/png;base64,/, "");
        const maskImage = new Image();

        maskImage.onload = () => {
            createOverlay();
            prepareMask(maskImage);
        };

        maskImage.onerror = () => {
            console.error("Inspector: could not load the class ID mask");
            clearHighlight();
        };

        maskImage.src = "data:image/png;base64," + cleanBase64;
        currentMaskUrl = maskImage.src;
    });
})();
