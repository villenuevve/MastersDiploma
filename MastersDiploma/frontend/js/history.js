(function () {
    "use strict";

    const STORAGE_KEY = "segmentationHistory";
    const MAX_ITEMS = 10;

    function readHistory() {
        try {
            const items = JSON.parse(
                localStorage.getItem(STORAGE_KEY) || "[]"
            );

            return Array.isArray(items)
                ? items.filter(item => item && typeof item === "object")
                : [];
        } catch {
            return [];
        }
    }

    function renderHistory() {
        const container = document.getElementById("historyList");
        if (!container) return;

        const items = readHistory();
        container.replaceChildren();

        if (items.length === 0) {
            const empty = document.createElement("p");
            empty.textContent =
                "No analyses yet. Run segmentation to build your history.";
            empty.style.color = "#667085";
            container.appendChild(empty);
            return;
        }

        items.forEach(item => {
            const card = document.createElement("div");
            card.style.cssText =
                "padding:14px 0;border-bottom:1px solid #e5e7eb;";

            const name = document.createElement("strong");
            name.textContent = String(item.filename || "image");

            const details = document.createElement("p");
            details.style.cssText =
                "margin-top:6px;font-size:13px;line-height:1.7;color:#667085;";

            const width = Number(item.width) || 0;
            const height = Number(item.height) || 0;
            const inference = Number(item.inference) || 0;

            details.textContent =
                `${String(item.date || "Unknown date")} · ` +
                `${width} × ${height} · ${inference.toFixed(2)} ms`;

            const dominant = document.createElement("p");
            dominant.style.cssText = "font-size:13px;margin-top:4px;";
            dominant.textContent =
                `Dominant class: ${String(item.dominantClass || "None")}`;

            card.append(name, details, dominant);
            container.appendChild(card);
        });
    }

    function addHistory(data) {
        if (!data) return;

        const classes = Array.isArray(data.detected_classes)
            ? data.detected_classes
            : [];

        const dominant = classes.reduce((best, item) => {
            if (!best) return item;

            return Number(item.percentage || 0) >
                Number(best.percentage || 0)
                ? item
                : best;
        }, null);

        const entry = {
            filename: String(data.filename || "image"),
            date: new Date().toLocaleString(),
            width: Number(data.original_width) || 0,
            height: Number(data.original_height) || 0,
            inference: Number(data.inference_time_ms) || 0,
            dominantClass: dominant
                ? String(dominant.name || "Unknown")
                : "None"
        };

        const items = [entry, ...readHistory()].slice(0, MAX_ITEMS);

        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
        } catch (error) {
            console.error("Unable to save segmentation history:", error);
        }

        renderHistory();
    }

    window.addEventListener("segmentation-result", event => {
        addHistory(event.detail);
    });

    document.addEventListener("DOMContentLoaded", renderHistory);
})();
