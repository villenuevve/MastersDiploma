(() => {
    "use strict";

    const elements = {
        total: document.getElementById("obstacleTotal"),
        dynamic: document.getElementById("obstacleDynamic"),
        static: document.getElementById("obstacleStatic"),
        obstacleList: document.getElementById("obstacleList"),
        environmentList: document.getElementById("environmentObjectList")
    };

    const image = document.getElementById("resultImage");

    if (!elements.obstacleList || !elements.environmentList) {
        console.error("Obstacle View: required HTML elements not found");
        return;
    }

    const CLASS_NAMES = {
        2: "Building",
        3: "Wall",
        4: "Fence",
        5: "Pole",
        6: "Traffic light",
        7: "Traffic sign",
        11: "Person",
        12: "Rider",
        13: "Car",
        14: "Truck",
        15: "Bus",
        16: "Train",
        17: "Motorcycle",
        18: "Bicycle"
    };

    function formatNumber(value) {
        return Number(value || 0).toLocaleString("en-US");
    }

    function formatPercentage(value) {
        const number = Number(value);

        return Number.isFinite(number)
            ? `${number.toFixed(2)}%`
            : "—";
    }

    function getClassName(item) {
        return item.class_name ||
            item.name ||
            CLASS_NAMES[item.class_id] ||
            `Class ${item.class_id}`;
    }

    function createEmptyMessage(message) {
        const paragraph = document.createElement("p");
        paragraph.className = "obstacle-empty";
        paragraph.textContent = message;

        return paragraph;
    }

    function selectObject(item, data) {
        if (!image || !item.bounding_box) return;

        const box = item.bounding_box;

        const centerX =
            (Number(box.x_min) + Number(box.x_max)) / 2;

        const centerY =
            (Number(box.y_min) + Number(box.y_max)) / 2;

        const originalWidth = Number(data.original_width);
        const originalHeight = Number(data.original_height);

        if (
            !originalWidth ||
            !originalHeight ||
            !image.complete ||
            !image.naturalWidth
        ) {
            return;
        }

        const rect = image.getBoundingClientRect();

        const clientX =
            rect.left + (centerX / originalWidth) * rect.width;

        const clientY =
            rect.top + (centerY / originalHeight) * rect.height;

        image.dispatchEvent(
            new MouseEvent("mousemove", {
                clientX,
                clientY,
                bubbles: true
            })
        );

        image.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });
    }

    function createObjectItem(item, data, type) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "obstacle-item";

        const main = document.createElement("span");
        main.className = "obstacle-item-main";

        const name = document.createElement("strong");
        name.textContent = getClassName(item);

        const details = document.createElement("span");
        details.className = "obstacle-item-details";

        const position = item.position || "unknown";

        details.textContent =
            `${position} · ${formatPercentage(item.area_percentage)} of image`;

        main.append(name, details);

        const meta = document.createElement("span");
        meta.className = "obstacle-item-meta";

        meta.textContent = `${formatNumber(item.area_pixels)} px`;

        const category = document.createElement("span");
        category.className = `obstacle-category ${type}`;
        category.textContent = type;

        button.append(main, meta, category);

        button.addEventListener("click", () => {
            selectObject(item, data);
        });

        return button;
    }

    function renderList(container, items, data, type, emptyText) {
        container.replaceChildren();

        if (!Array.isArray(items) || items.length === 0) {
            container.appendChild(createEmptyMessage(emptyText));
            return;
        }

        const fragment = document.createDocumentFragment();

        items.forEach(item => {
            fragment.appendChild(
                createObjectItem(item, data, type)
            );
        });

        container.appendChild(fragment);
    }

    function render(data) {
        const summary = data.obstacle_summary || {};

        const obstacles = Array.isArray(data.obstacles)
            ? data.obstacles
            : [];

        const environmentObjects =
            Array.isArray(data.environment_objects)
                ? data.environment_objects
                : [];

        if (elements.total) {
            elements.total.textContent =
                formatNumber(summary.total ?? obstacles.length);
        }

        if (elements.dynamic) {
            elements.dynamic.textContent =
                formatNumber(summary.dynamic);
        }

        if (elements.static) {
            elements.static.textContent =
                formatNumber(summary.static);
        }

        const dynamicObstacles = obstacles.filter(
            item => item.category === "dynamic"
        );

        const staticObstacles = obstacles.filter(
            item => item.category === "static"
        );

        elements.obstacleList.replaceChildren();

        if (obstacles.length === 0) {
            elements.obstacleList.appendChild(
                createEmptyMessage("No potential obstacles detected.")
            );
        } else {
            const dynamicHeading = document.createElement("h4");
            dynamicHeading.className = "obstacle-group-title";
            dynamicHeading.textContent =
                `Dynamic objects (${dynamicObstacles.length})`;

            elements.obstacleList.appendChild(dynamicHeading);

            if (dynamicObstacles.length === 0) {
                elements.obstacleList.appendChild(
                    createEmptyMessage("No dynamic objects detected.")
                );
            } else {
                dynamicObstacles.forEach(item => {
                    elements.obstacleList.appendChild(
                        createObjectItem(item, data, "dynamic")
                    );
                });
            }

            const staticHeading = document.createElement("h4");
            staticHeading.className = "obstacle-group-title";
            staticHeading.textContent =
                `Static objects (${staticObstacles.length})`;

            elements.obstacleList.appendChild(staticHeading);

            if (staticObstacles.length === 0) {
                elements.obstacleList.appendChild(
                    createEmptyMessage("No static objects detected.")
                );
            } else {
                staticObstacles.forEach(item => {
                    elements.obstacleList.appendChild(
                        createObjectItem(item, data, "static")
                    );
                });
            }
        }

        renderList(
            elements.environmentList,
            environmentObjects,
            data,
            "environment",
            "No environment objects detected."
        );
    }

    window.addEventListener("segmentation-view-selected", event => {
        render(event.detail || {});
    });

    console.log("Obstacle View initialized");
})();
