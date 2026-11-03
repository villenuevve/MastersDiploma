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

    function createElement(tag, text, styles = "") {
        const element = document.createElement(tag);

        if (text !== undefined && text !== null) {
            element.textContent = String(text);
        }

        if (styles) {
            element.style.cssText = styles;
        }

        return element;
    }

    function renderHistory() {
        const container = document.getElementById("historyList");
        if (!container) return;

        const items = readHistory();
        container.replaceChildren();

        if (items.length === 0) {
            container.appendChild(
                createElement(
                    "p",
                    "No analyses yet. Run segmentation to build your history.",
                    "color:#667085;"
                )
            );
            return;
        }

        items.forEach((item, index) => {
            const card = createElement(
                "div",
                undefined,
                "padding:16px 0;border-bottom:1px solid #e5e7eb;"
            );

            const header = createElement(
                "div",
                undefined,
                "display:flex;justify-content:space-between;" +
                "align-items:center;gap:12px;flex-wrap:wrap;"
            );

            header.appendChild(
                createElement(
                    "strong",
                    `${index + 1}. ${item.filename || "image"}`,
                    "font-size:15px;"
                )
            );

            header.appendChild(
                createElement(
                    "span",
                    item.date || "Unknown date",
                    "font-size:12px;color:#667085;"
                )
            );

            const details = createElement(
                "p",
                undefined,
                "margin-top:8px;font-size:13px;line-height:1.8;" +
                "color:#667085;"
            );

            const width = Number(item.width) || 0;
            const height = Number(item.height) || 0;
            const inference = Number(item.inference) || 0;

            details.textContent =
                `Resolution: ${width} × ${height} · ` +
                `Inference: ${inference.toFixed(2)} ms`;

            const model = createElement(
                "p",
                `Model: ${item.model || "Unknown"} · ` +
                `Device: ${item.device || "Unknown"}`,
                "font-size:13px;color:#667085;margin-top:3px;" +
                "overflow-wrap:anywhere;"
            );

            const classes = Array.isArray(item.classes)
                ? item.classes
                : [];

            const summary = createElement(
                "p",
                `Detected classes: ${item.classCount ?? classes.length} · ` +
                `Dominant class: ${item.dominantClass || "None"}`,
                "font-size:13px;margin-top:5px;"
            );

            card.append(header, details, model, summary);

            if (classes.length > 0) {
                const toggle = createElement(
                    "button",
                    "Show detected classes",
                    "margin-top:8px;padding:7px 12px;border:1px solid " +
                    "#667085;border-radius:6px;background:#344054;" +
                    "color:#ffffff;cursor:pointer;font-size:12px;"
                );

                const classList = createElement(
                    "div",
                    undefined,
                    "display:none;margin-top:10px;padding:10px 12px;" +
                    "background:#f8fafc;border-radius:8px;"
                );

                classes.forEach(item => {
                    const row = createElement(
                        "div",
                        `${item.name || "Unknown"} — ` +
                        `${Number(item.percentage || 0).toFixed(2)}%`,
                        "font-size:12px;padding:4px 0;color:#475467;"
                    );

                    classList.appendChild(row);
                });

                toggle.addEventListener("click", () => {
                    const isHidden = classList.style.display === "none";

                    classList.style.display = isHidden ? "block" : "none";
                    toggle.textContent = isHidden
                        ? "Hide detected classes"
                        : "Show detected classes";
                });

                card.append(toggle, classList);
            }

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
            model: String(data.model || "Unknown"),
            device: String(data.device || "Unknown"),
            classCount: classes.length,
            dominantClass: dominant
                ? String(dominant.name || "Unknown")
                : "None",
            classes: classes.map(item => ({
                name: String(item.name || "Unknown"),
                percentage: Number(item.percentage) || 0
            }))
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
