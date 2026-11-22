    const API_URL =
        "http://127.0.0.1:8000";


    /* =================
       CITYSCAPES COLORS
    ======================= */

    const CITYSCAPES_COLORS = [

        [128, 64, 128],   // road
        [244, 35, 232],   // sidewalk
        [70, 70, 70],     // building
        [102, 102, 156],  // wall
        [190, 153, 153],  // fence
        [153, 153, 153],  // pole
        [250, 170, 30],   // traffic light
        [220, 220, 0],    // traffic sign
        [107, 142, 35],   // vegetation
        [152, 251, 152],  // terrain
        [70, 130, 180],   // sky
        [220, 20, 60],    // person
        [255, 0, 0],      // rider
        [0, 0, 142],      // car
        [0, 0, 70],       // truck
        [0, 60, 100],     // bus
        [0, 80, 100],     // train
        [0, 0, 230],      // motorcycle
        [119, 11, 32]     // bicycle

    ];


    /* ============
       DOM
    =========== */

    const fileInput =
        document.getElementById("fileInput");

    const uploadArea =
        document.getElementById("uploadArea");

    const selectedFile =
        document.getElementById("selectedFile");

    const predictButton =
        document.getElementById("predictButton");

    const loading =
        document.getElementById("loading");

    const errorBox =
        document.getElementById("error");

    const resultSection =
        document.getElementById("resultSection");

    const originalImage =
        document.getElementById("originalImage");

    const resultImage =
        document.getElementById("resultImage");

    const resultTitle =
        document.getElementById("resultTitle");

    const comparisonImage2 =
        document.getElementById("comparisonImage2");

    const comparisonTitle2 =
        document.getElementById("comparisonTitle2");

    const comparisonSelect1 =
        document.getElementById("comparisonSelect1");

    const comparisonSelect2 =
        document.getElementById("comparisonSelect2");

    const overlayButton =
        document.getElementById("overlayButton");

    const maskButton =
        document.getElementById("maskButton");

    const traversabilityButton =
        document.getElementById("traversabilityButton");

    const confidenceButton = document.getElementById("confidenceButton");

    const depthButton = document.getElementById("depthButton") || (() => {
        const button = document.createElement("button");
        button.id = "depthButton";
        button.type = "button";
        button.textContent = "Depth Map";
        button.className = document.getElementById("overlayButton")?.className || "";
        button.disabled = true;
        const reference = document.getElementById("overlayButton");
        (reference?.parentElement || document.body).appendChild(button);
        return button;
    })();
    const uncertaintyButton = document.getElementById("uncertaintyButton") || (() => {
        const button = document.createElement("button");
        button.id = "uncertaintyButton";
        button.type = "button";
        button.textContent = "Uncertainty Map";
        button.className = document.getElementById("overlayButton")?.className || "";
        button.disabled = true;
        const reference = document.getElementById("overlayButton");
        (reference?.parentElement || document.body).appendChild(button);
        return button;
    })();
    const downloadDepth = document.getElementById("downloadDepth") || (() => {
        const link = document.createElement("a");
        link.id = "downloadDepth";
        link.className = "download-button secondary";
        link.textContent = "Download Depth Map";
        link.style.display = "none";
        const reference = document.getElementById("downloadOverlay");
        (reference?.parentElement || document.body).appendChild(link);
        return link;
    })();
    const downloadUncertainty = document.getElementById("downloadUncertainty") || (() => {
        const link = document.createElement("a");
        link.id = "downloadUncertainty";
        link.className = "download-button secondary";
        link.textContent = "Download Uncertainty Map";
        link.style.display = "none";
        const reference = document.getElementById("downloadOverlay");
        (reference?.parentElement || document.body).appendChild(link);
        return link;
    })();

    const riskButton = document.getElementById("riskButton");
    const downloadRiskMap = document.getElementById("downloadRiskMap");
    const riskLegend = document.getElementById("riskLegend");
    const riskMetrics = document.getElementById("riskMetrics");

    const safeCorridorButton =
        document.getElementById("safeCorridorButton");
    const downloadSafeCorridor =
        document.getElementById("downloadSafeCorridor");
    const safeCorridorLegend =
        document.getElementById("safeCorridorLegend");
    const safeCorridorMetrics =
        document.getElementById("safeCorridorMetrics");

    const downloadConfidence = document.getElementById("downloadConfidence");
    const confidenceMetrics = document.getElementById("confidenceMetrics");
    const meanConfidence = document.getElementById("meanConfidence");
    const lowConfidence = document.getElementById("lowConfidence");
    const mediumConfidence = document.getElementById("mediumConfidence");
    const highConfidence = document.getElementById("highConfidence");

    const confidenceLowThresholdInput =
        document.getElementById("confidenceLowThreshold");
    const confidenceHighThresholdInput =
        document.getElementById("confidenceHighThreshold");

    const confidenceLowThresholdValue =
        document.getElementById("confidenceLowThresholdValue");
    const confidenceHighThresholdValue =
        document.getElementById("confidenceHighThresholdValue");

    const confidenceLowLegend =
        document.getElementById("confidenceLowLegend");
    const confidenceMediumLegend =
        document.getElementById("confidenceMediumLegend");
    const confidenceHighLegend =
        document.getElementById("confidenceHighLegend");

    const recalculateConfidenceButton =
        document.getElementById("recalculateConfidence");
    const confidenceStatus =
        document.getElementById("confidenceStatus");

    const downloadTraversability =
        document.getElementById("downloadTraversability");

    const traversabilityLegend =
        document.getElementById("traversabilityLegend");

    const obstacleRadiusInput =
        document.getElementById("obstacleRadius");
    const obstacleRadiusValue =
        document.getElementById("obstacleRadiusValue");
    const recalculateTraversabilityButton =
        document.getElementById("recalculateTraversability");
    const traversabilityStatus =
        document.getElementById("traversabilityStatus");

    const confidenceLegend =
        document.getElementById("confidenceLegend");

    const downloadOverlay =
        document.getElementById("downloadOverlay");

    const downloadMask =
        document.getElementById("downloadMask");

    const downloadArchive =
        document.getElementById("downloadArchive");

    const classesContainer =
        document.getElementById("classes");

    const classCount =
        document.getElementById("classCount");


    let selectedImage = null;

    let overlayUrl = "";

    let maskUrl = "";

    let traversabilityUrl = "";
    let confidenceUrl = "";
    let confidenceFilename = "";
    let depthUrl = "";
    let uncertaintyUrl = "";
    let riskMapUrl = "";
    let safeCorridorUrl = "";
        let classIdsFilename = "";


    /* =============
       FILE INPUT
    ================ */


    let selectedImages = [];

    const batchProgress = document.getElementById("batchProgress");
    const batchResults = document.getElementById("batchResults");
    const batchSummary = document.getElementById("batchSummary");
    const batchResultsGrid = document.getElementById("batchResultsGrid");

    const originalPredictButtonText = predictButton.textContent.trim();

    function isSupportedImage(file) {
        return file &&
            (
                (file.type && file.type.startsWith("image/")) ||
                /\.(jpe?g|png|webp|bmp|tiff?)$/i.test(file.name || "")
            );
    }

    function handleFiles(fileList) {
        const files = Array.from(fileList || []);
        if (!files.length) return;

        const validFiles = files.filter(isSupportedImage);
        const rejected = files.length - validFiles.length;

        if (!validFiles.length) {
            showError("Please select valid image files.");
            return;
        }

        selectedImages = validFiles;
        selectImage(validFiles[0]);

        selectedFile.textContent =
            `${validFiles.length} image(s) selected` +
            (rejected ? ` · ${rejected} unsupported file(s) skipped` : "");

        predictButton.textContent = validFiles.length > 1
            ? `Process ${validFiles.length} images`
            : originalPredictButtonText;

        batchProgress.style.display = "none";
        batchResults.style.display = "none";
        batchResultsGrid.replaceChildren();
        batchSummary.textContent = "";
    }

    fileInput.addEventListener("change", function () {
        handleFiles(this.files);
        this.value = "";
    });

    function selectImage(file) {
        if (!isSupportedImage(file)) {
            showError("Please select a valid image file.");
            return;
        }

        selectedImage = file;

        selectedFile.style.display = "block";
        selectedFile.textContent = `Selected file: ${file.name}`;

        predictButton.disabled = false;
        errorBox.style.display = "none";

        window.SegmentationView.prepareForNewImage(
            originalImage,
            resultImage,
            resultSection,
            classesContainer,
            classCount,
            file
        );

        overlayUrl = "";
        maskUrl = "";
        traversabilityUrl = "";
        confidenceUrl = "";
        confidenceFilename = "";
        depthUrl = "";
        uncertaintyUrl = "";
        classIdsFilename = "";

        depthButton.disabled = true;
        uncertaintyButton.disabled = true;
        downloadDepth.style.display = "none";
        downloadUncertainty.style.display = "none";

        depthButton.classList.remove("active");
        uncertaintyButton.classList.remove("active");

        confidenceStatus.textContent =
            "Select an image to calculate its confidence map.";

        comparisonSelect1.value = "overlay";
        comparisonSelect2.value = "mask";

        comparisonImage2.removeAttribute("src");
        comparisonTitle2.textContent = "Segmentation mask";

        obstacleRadiusInput.value = "12";
        obstacleRadiusValue.textContent = "12";
        traversabilityStatus.textContent =
            "Adjust the radius and recalculate the map.";

        resetTraversabilityStatistics();

        traversabilityLegend.style.display = "none";
        confidenceLegend.style.display = "none";
        confidenceMetrics.style.display = "none";
        traversabilityButton.disabled = true;
        confidenceButton.disabled = true;

        overlayButton.classList.remove("active");
        maskButton.classList.remove("active");
    }

    uploadArea.addEventListener("dragover", function (event) {
        event.preventDefault();
        uploadArea.classList.add("dragover");
    });

    uploadArea.addEventListener("dragleave", function () {
        uploadArea.classList.remove("dragover");
    });

    uploadArea.addEventListener("drop", function (event) {
        event.preventDefault();
        uploadArea.classList.remove("dragover");
        handleFiles(event.dataTransfer.files);
    });

    function dispatchSegmentationViewSelected(data) {
        window.dispatchEvent(
            new CustomEvent("segmentation-view-selected", { detail: data })
        );
    }

    function dispatchSegmentationHistory(data) {
        window.dispatchEvent(
            new CustomEvent("segmentation-result", { detail: data })
        );

        dispatchSegmentationViewSelected(data);
    }

    function addBatchResultCard(file, data, error) {
        const card = document.createElement("article");
        card.className = "batch-result-card";

        const heading = document.createElement("strong");
        heading.textContent = file.name;
        card.appendChild(heading);

        const status = document.createElement("div");
        status.className = "batch-result-status";
        status.textContent = error
            ? `Failed: ${error}`
            : "Successfully processed";
        card.appendChild(status);

        if (data) {
            const details = document.createElement("p");
            details.textContent =
                `${data.original_width || "—"} × ${data.original_height || "—"} px` +
                ` · ${Number(data.inference_time_ms || 0).toFixed(2)} ms`;
            card.appendChild(details);

            const links = document.createElement("div");
            links.className = "batch-result-links";

            function addLink(label, url) {
                if (!url) return;

                const link = document.createElement("a");
                link.className = "download-button secondary";
                link.textContent = `↓ ${label}`;
                link.href = url;
                link.download = "";
                links.appendChild(link);
            }

            function downloadUrl(url) {
                if (!url) return "";
                return `${url}${url.includes("?") ? "&" : "?"}download=true`;
            }

            const overlayName = data.overlay_file
                ? data.overlay_file.split("/").pop()
                : "";

            const maskName = data.mask_file
                ? data.mask_file.split("/").pop()
                : "";

            const overlayFullUrl = overlayName
                ? `${API_URL}/results/${overlayName}`
                : "";

            const maskFullUrl = maskName
                ? `${API_URL}/results/${maskName}`
                : "";

            const traversalUrl = data.traversability_url
                ? `${API_URL}${data.traversability_url}`
                : (
                    data.traversability_file
                        ? `${API_URL}/results/${data.traversability_file.split("/").pop()}`
                        : ""
                );

            const confidenceResultUrl = data.confidence_url
                ? `${API_URL}${data.confidence_url}`
                : "";

            const depthResultUrl = data.depth_url
                ? `${API_URL}${data.depth_url}`
                : (
                    data.depth_file
                        ? `${API_URL}/results/${data.depth_file.split("/").pop()}`
                        : ""
                );

            const uncertaintyResultUrl = data.uncertainty_url
                ? `${API_URL}${data.uncertainty_url}`
                : (
                    data.uncertainty_file
                        ? `${API_URL}/results/${data.uncertainty_file.split("/").pop()}`
                        : ""
                );

            const riskResultUrl = data.risk_map_url
                ? `${API_URL}${data.risk_map_url}`
                : (
                    data.risk_map_file
                        ? `${API_URL}/results/${data.risk_map_file.split("/").pop()}`
                        : ""
                );

            const safeCorridorResultUrl = data.safe_corridor_url
                ? `${API_URL}${data.safe_corridor_url}`
                : (
                    data.safe_corridor_file
                        ? `${API_URL}/results/${data.safe_corridor_file.split("/").pop()}`
                        : ""
                );

            if (overlayFullUrl) {
                const overlayPreview = document.createElement("img");
                overlayPreview.className = "batch-overlay-preview";
                overlayPreview.alt = "Segmentation overlay preview";
                overlayPreview.style.width = "100%";
                overlayPreview.style.maxHeight = "180px";
                overlayPreview.style.objectFit = "contain";
                overlayPreview.style.marginTop = "10px";
                overlayPreview.style.borderRadius = "8px";

                window.Auth.loadProtectedImage(
                    overlayPreview,
                    overlayFullUrl
                );

                card.appendChild(overlayPreview);
            }

            addLink("Overlay", downloadUrl(overlayFullUrl));
            addLink("Mask", downloadUrl(maskFullUrl));
            addLink("Traversability", downloadUrl(traversalUrl));
            addLink("Confidence", downloadUrl(confidenceResultUrl));
            addLink("Depth Map", downloadUrl(depthResultUrl));
            addLink("Uncertainty Map", downloadUrl(uncertaintyResultUrl));
            addLink("Risk Map", downloadUrl(riskResultUrl));
            addLink("Safe Corridor", downloadUrl(safeCorridorResultUrl));

            if (data.upload_id && data.class_ids_filename) {
                const archiveUrl =
                    `${API_URL}/results/archive/${encodeURIComponent(data.class_ids_filename)}` +
                    `?upload_id=${encodeURIComponent(data.upload_id)}`;

                addLink("Download ZIP", archiveUrl);
            }

            card.appendChild(links);
        }

        // Make the entire result card open the full analysis.
        if (data && !error) {
            card.classList.add("batch-result-card-clickable");
            card.setAttribute("role", "button");
            card.setAttribute("tabindex", "0");
            card.setAttribute("title", "Click to view full analysis");

            const openResult = async () => {
                if (predictButton.disabled) return;

                // Інвалідуємо попередні асинхронні операції відображення.
                displayRequestId++;

                selectedImage = file;
                selectedImages = [file];

                selectedFile.style.display = "block";
                selectedFile.textContent = `Selected file: ${file.name}`;

                predictButton.textContent = originalPredictButtonText;

                batchResultsGrid
                    .querySelectorAll(".batch-result-card")
                    .forEach(item => item.classList.remove("is-selected"));

                card.classList.add("is-selected");

                errorBox.style.display = "none";

                try {
                    await displayPredictionResult(data, file);
                    dispatchSegmentationViewSelected(data);
                } catch (error) {
                    console.error("Unable to display batch result:", error);
                    showError(error.message || String(error));
                }
            };

            card.addEventListener("click", event => {
                if (event.target.closest("a, button")) return;
                openResult();
            });

            card.addEventListener("keydown", event => {
                if (
                    event.target === card &&
                    (event.key === "Enter" || event.key === " ")
                ) {
                    event.preventDefault();
                    openResult();
                }
            });
        }

        batchResultsGrid.appendChild(card);
    }

    let displayRequestId = 0;


    function renderSceneAnalysis(sceneData) {
        let section = document.getElementById("scene-analysis");

        if (!sceneData) {
            if (section) section.style.display = "none";
            return;
        }

        if (!section) {
            section = document.createElement("section");
            section.id = "scene-analysis";
            section.style.cssText = [
                "margin:24px 0",
                "padding:20px",
                "border:1px solid var(--border-color,#344b50)",
                "border-radius:14px",
                "background:var(--card-background,transparent)"
            ].join(";");

            const heading = document.createElement("h2");
            heading.textContent = "Scene Analysis";
            heading.style.marginTop = "0";

            const description = document.createElement("p");
            description.textContent =
                "Detected objects, their image-space positions and navigation surfaces.";
            description.style.opacity = "0.7";

            const metrics = document.createElement("div");
            metrics.id = "scene-analysis-metrics";
            metrics.style.cssText =
                "display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin:18px 0;";

            const tableContainer = document.createElement("div");
            tableContainer.id = "scene-analysis-objects";
            tableContainer.style.cssText = "overflow-x:auto;";

            section.append(heading, description, metrics, tableContainer);
        }

        section.style.display = "block";

        const metrics = section.querySelector("#scene-analysis-metrics");
        const tableContainer =
            section.querySelector("#scene-analysis-objects");

        const navigation = sceneData.navigation_surface || {};

        function metric(label, value) {
            const card = document.createElement("div");
            card.style.cssText =
                "padding:14px;border:1px solid var(--border-color,#344b50);border-radius:10px;";

            const title = document.createElement("div");
            title.textContent = label;
            title.style.cssText = "font-size:12px;opacity:.7;margin-bottom:6px;";

            const number = document.createElement("strong");
            number.textContent = value;

            card.append(title, number);
            return card;
        }

        metrics.replaceChildren(
            metric("Detected objects", String(sceneData.object_count ?? 0)),
            metric("Potential obstacles", String(sceneData.obstacle_count ?? 0)),
            metric(
                "Road surface",
                `${Number(navigation.road_percentage ?? 0).toFixed(2)}%`
            ),
            metric(
                "Sidewalk",
                `${Number(navigation.sidewalk_percentage ?? 0).toFixed(2)}%`
            )
        );

        tableContainer.replaceChildren();

        const objects = Array.isArray(sceneData.objects)
            ? sceneData.objects
            : [];

        const subheading = document.createElement("h3");
        subheading.textContent = "Detected objects";
        tableContainer.appendChild(subheading);

        if (!objects.length) {
            const empty = document.createElement("p");
            empty.textContent = "No objects detected.";
            tableContainer.appendChild(empty);
        } else {
            const table = document.createElement("table");
            table.style.cssText =
                "width:100%;border-collapse:collapse;text-align:left;";

            const headers = [
                "Class", "Position", "Area (px)", "Confidence"
            ];

            const thead = document.createElement("thead");
            const headerRow = document.createElement("tr");

            headers.forEach(label => {
                const th = document.createElement("th");
                th.textContent = label;
                th.style.cssText =
                    "padding:10px;border-bottom:1px solid var(--border-color,#344b50);";
                headerRow.appendChild(th);
            });

            thead.appendChild(headerRow);
            table.appendChild(thead);

            const tbody = document.createElement("tbody");

            objects.forEach(object => {
                const row = document.createElement("tr");

                let confidence = Number(object.mean_confidence ?? 0);
                if (confidence <= 1) confidence *= 100;

                const values = [
                    object.class_name || "Unknown",
                    object.position || "—",
                    Number(object.area_pixels ?? 0).toLocaleString(),
                    `${confidence.toFixed(1)}%`
                ];

                values.forEach(value => {
                    const cell = document.createElement("td");
                    cell.textContent = String(value);
                    cell.style.cssText =
                        "padding:10px;border-bottom:1px solid var(--border-color,#344b50);";
                    row.appendChild(cell);
                });

                tbody.appendChild(row);
            });

            table.appendChild(tbody);
            tableContainer.appendChild(table);
        }

        if (classesContainer && classesContainer.parentElement) {
            classesContainer.insertAdjacentElement("afterend", section);
        } else {
            resultSection.appendChild(section);
        }
    }

    function renderRiskStatistics(statistics) {
        if (!statistics) {
            riskMetrics.style.display = "none";
            return;
        }

        const formatPercentage = value =>
            `${Number(value || 0).toFixed(2)}%`;

        document.getElementById("riskLowPercentage").textContent =
            formatPercentage(statistics.low_risk_percentage);

        document.getElementById("riskMediumPercentage").textContent =
            formatPercentage(statistics.medium_risk_percentage);

        document.getElementById("riskHighPercentage").textContent =
            formatPercentage(statistics.high_risk_percentage);

        document.getElementById("meanRisk").textContent =
            Number(statistics.mean_risk || 0).toFixed(2);

        document.getElementById("maxRisk").textContent =
            `${Number(statistics.max_risk || 0).toFixed(0)} / 100`;

        riskMetrics.style.display = "block";
    }

    function renderSafeCorridorStatistics(statistics) {
        if (!statistics) {
            safeCorridorMetrics.style.display = "none";
            return;
        }

        document.getElementById("safeCorridorFound").textContent =
            statistics.found ? "Yes" : "No";

        document.getElementById("corridorPercentage").textContent =
            `${Number(statistics.corridor_percentage || 0).toFixed(2)}%`;

        document.getElementById("corridorPixels").textContent =
            `${Number(statistics.corridor_pixels || 0).toLocaleString()} px`;

        document.getElementById("meanTraversability").textContent =
            Number(statistics.mean_traversability || 0).toFixed(2);

        document.getElementById("safeCorridorMeanRisk").textContent =
            Number(statistics.mean_risk || 0).toFixed(2);

        document.getElementById("safeCorridorDirection").textContent =
            statistics.direction || "—";

        safeCorridorMetrics.style.display = "block";
    }


    async function displayPredictionResult(data, sourceFile, savedResult = false) {
        const requestId = ++displayRequestId;
        const imageForDisplay = sourceFile || selectedImage;
        const overlayFilename = data.overlay_file.split("/").pop();
        const maskFilename = data.mask_file.split("/").pop();

        const fullOverlayUrl =
            `${API_URL}/results/${overlayFilename}`;

        // Always load the unique full-resolution overlay for this image.
        overlayUrl = fullOverlayUrl;

        maskUrl = `${API_URL}/results/${maskFilename}`;

        const traversabilityFilename = data.traversability_file
            ? data.traversability_file.split("/").pop()
            : "";

        traversabilityUrl = data.traversability_url
            ? `${API_URL}${data.traversability_url}`
            : (
                traversabilityFilename
                    ? `${API_URL}/results/${traversabilityFilename}`
                    : ""
            );

        riskMapUrl = data.risk_map_url
            ? `${API_URL}${data.risk_map_url}`
            : (
                data.risk_map_file
                    ? `${API_URL}/results/${data.risk_map_file.split("/").pop()}`
                    : ""
            );

        safeCorridorUrl = data.safe_corridor_url
            ? `${API_URL}${data.safe_corridor_url}`
            : (
                data.safe_corridor_file
                    ? `${API_URL}/results/${data.safe_corridor_file.split("/").pop()}`
                    : ""
            );

        safeCorridorButton.disabled = !safeCorridorUrl;
        downloadSafeCorridor.style.display =
            safeCorridorUrl ? "inline-flex" : "none";

        if (safeCorridorUrl) {
            const safeCorridorFilename = data.safe_corridor_file
                ? data.safe_corridor_file.split("/").pop()
                : data.safe_corridor_url.split("/").pop();

            downloadSafeCorridor.href =
                `${safeCorridorUrl}${safeCorridorUrl.includes("?") ? "&" : "?"}download=true`;
            downloadSafeCorridor.download = safeCorridorFilename;
        }

        renderSafeCorridorStatistics(data.safe_corridor_statistics);

        riskButton.disabled = !riskMapUrl;
        downloadRiskMap.style.display = riskMapUrl ? "inline-flex" : "none";

        if (riskMapUrl) {
            const riskFilename = data.risk_map_file
                ? data.risk_map_file.split("/").pop()
                : "risk_map.png";

            downloadRiskMap.href =
                `${riskMapUrl}${riskMapUrl.includes("?") ? "&" : "?"}download=true`;
            downloadRiskMap.download = riskFilename;
        }

        renderRiskStatistics(data.risk_statistics);

        classIdsFilename = data.class_ids_filename || "";

        if (data.upload_id && classIdsFilename) {
            downloadArchive.href =
                `${API_URL}/results/archive/${encodeURIComponent(classIdsFilename)}` +
                `?upload_id=${encodeURIComponent(data.upload_id)}`;
            downloadArchive.style.display = "inline-flex";
        } else {
            downloadArchive.style.display = "none";
        }

        traversabilityButton.disabled = !traversabilityUrl;

        confidenceUrl = data.confidence_url
            ? `${API_URL}${data.confidence_url}`
            : "";

        depthUrl = data.depth_url
            ? `${API_URL}${data.depth_url}`
            : (
                data.depth_file
                    ? `${API_URL}/results/${data.depth_file.split("/").pop()}`
                    : ""
            );

        uncertaintyUrl = data.uncertainty_url
            ? `${API_URL}${data.uncertainty_url}`
            : (
                data.uncertainty_file
                    ? `${API_URL}/results/${data.uncertainty_file.split("/").pop()}`
                    : ""
            );

        depthButton.disabled = !depthUrl;
        uncertaintyButton.disabled = !uncertaintyUrl;

        downloadDepth.style.display = depthUrl ? "inline-flex" : "none";
        downloadUncertainty.style.display =
            uncertaintyUrl ? "inline-flex" : "none";

        if (depthUrl) {
            downloadDepth.href =
                `${depthUrl}${depthUrl.includes("?") ? "&" : "?"}download=true`;
            downloadDepth.download =
                data.depth_file
                    ? data.depth_file.split("/").pop()
                    : depthUrl.split("/").pop();
        }

        if (uncertaintyUrl) {
            downloadUncertainty.href =
                `${uncertaintyUrl}${uncertaintyUrl.includes("?") ? "&" : "?"}download=true`;
            downloadUncertainty.download =
                data.uncertainty_file
                    ? data.uncertainty_file.split("/").pop()
                    : uncertaintyUrl.split("/").pop();
        }

        confidenceButton.disabled = !confidenceUrl;
        downloadConfidence.style.display =
            confidenceUrl ? "inline-flex" : "none";

        confidenceFilename = data.confidence_file
            ? data.confidence_file.split("/").pop()
            : "";

        if (confidenceUrl) {
            const downloadFilename = confidenceFilename
                ? confidenceFilename.replace(/\.npy$/i, ".png")
                : "confidence.png";

            downloadConfidence.href =
                `${confidenceUrl}${confidenceUrl.includes("?") ? "&" : "?"}download=true`;
            downloadConfidence.download = downloadFilename;

            confidenceMetrics.style.display = "none";

            if (confidenceFilename && !savedResult) {
                await recalculateConfidence(true, requestId);
                if (requestId !== displayRequestId) return;
            } else if (savedResult) {
                meanConfidence.textContent =
                    `${data.mean_confidence ?? "—"}%`;
                lowConfidence.textContent =
                    `${data.low_confidence_percentage ?? "—"}%`;
                confidenceMetrics.style.display = "block";
                confidenceStatus.textContent =
                    "Loaded saved confidence map.";
            } else {
                meanConfidence.textContent =
                    `${data.mean_confidence}%`;

                lowConfidence.textContent =
                    `${data.low_confidence_percentage}%`;

                confidenceStatus.textContent =
                    "Recalculation unavailable: confidence data filename was not returned by the API.";
            }
        } else {
            confidenceMetrics.style.display = "none";
            confidenceFilename = "";
        }

        if (traversabilityUrl) {
            downloadTraversability.href =
                `${traversabilityUrl}${traversabilityUrl.includes("?") ? "&" : "?"}download=true`;
            downloadTraversability.download = traversabilityFilename;
            traversabilityLegend.style.display = "none";
        } else {
            traversabilityLegend.style.display = "none";
        }

        if (classIdsFilename && traversabilityUrl && !savedResult) {
            await recalculateTraversability(true, requestId);
            if (requestId !== displayRequestId) return;
        } else if (savedResult && traversabilityUrl) {
            traversabilityStatus.textContent =
                "Loaded saved traversability map.";
        } else {
            traversabilityStatus.textContent =
                "Recalculation unavailable: class-ID filename was not returned by the API.";
        }

        if (requestId !== displayRequestId) return;

        window.SegmentationView.prepareForNewImage(
            originalImage,
            resultImage,
            resultSection,
            classesContainer,
            classCount,
            imageForDisplay
        );

        window.SegmentationView.showImage(
            resultImage,
            overlayUrl
        );

        resultTitle.textContent = "Segmentation overlay";

        comparisonSelect1.value = "overlay";
        comparisonSelect2.value = "mask";

        updateComparisonOptions();
        renderComparisonPanel2("mask");

        overlayButton.classList.add("active");
        maskButton.classList.remove("active");
        traversabilityButton.classList.remove("active");
        confidenceButton.classList.remove("active");
        riskButton.classList.remove("active");
        safeCorridorButton.classList.remove("active");
        riskLegend.style.display = "none";
        riskMetrics.style.display = "none";
        safeCorridorLegend.style.display = "none";
        safeCorridorMetrics.style.display = "none";

        downloadOverlay.href =
            `${fullOverlayUrl}${fullOverlayUrl.includes("?") ? "&" : "?"}download=true`;
        downloadOverlay.download = overlayFilename;

        downloadMask.href =
            `${maskUrl}${maskUrl.includes("?") ? "&" : "?"}download=true`;
        downloadMask.download = maskFilename;

        window.SegmentationView.renderImageInformation(data);

        window.SegmentationView.renderDetectedClasses(
            data.detected_classes,
            classesContainer,
            classCount,
            CITYSCAPES_COLORS
        );

        renderSceneAnalysis(data.scene_analysis);

        window.SegmentationView.showResultSection(resultSection);

        resultSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }

    window.openSavedPredictionResult = async function(data, sourceFile) {
        await displayPredictionResult(data, sourceFile, true);

        const inspectorData = { ...data };

        // Saved metadata omits the Base64 mask; load it through the protected API.
        if (!inspectorData.class_ids_base64 && inspectorData.class_ids_filename) {
            try {
                const maskResponse = await window.Auth.apiFetch(
                    `${API_URL}/results/${encodeURIComponent(inspectorData.class_ids_filename)}`
                );

                if (!maskResponse.ok) {
                    throw new Error(
                        `Could not load class ID mask: HTTP ${maskResponse.status}`
                    );
                }

                const maskBlob = await maskResponse.blob();
                const maskDataUrl = await new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onload = () => resolve(reader.result);
                    reader.onerror = () => reject(
                        reader.error || new Error('Could not read class ID mask.')
                    );
                    reader.readAsDataURL(maskBlob);
                });

                inspectorData.class_ids_base64 = String(maskDataUrl).split(',', 2)[1];
            } catch (error) {
                console.error('Inspector: could not load saved class ID mask:', error);
            }
        }

        dispatchSegmentationViewSelected(inspectorData);
    };

    async function runBatchSegmentation(files) {
        let succeeded = 0;
        let failed = 0;
        let lastSuccessfulData = null;
        let lastSuccessfulFile = null;

        batchProgress.style.display = "block";
        batchResults.style.display = "block";
        batchResultsGrid.replaceChildren();

        for (let index = 0; index < files.length; index++) {
            const file = files[index];

            batchProgress.textContent =
                `Processing ${index + 1} / ${files.length}: ${file.name}`;

            try {
                const data = await window.SegmentationModel.predict(
                    file,
                    API_URL
                );

                succeeded++;
                lastSuccessfulData = data;
                lastSuccessfulFile = file;

                dispatchSegmentationHistory(data);
                addBatchResultCard(file, data, null);
            } catch (error) {
                failed++;
                console.error(`Batch processing failed for ${file.name}:`, error);
                addBatchResultCard(
                    file,
                    null,
                    error.message || String(error)
                );
            }

            batchSummary.textContent =
                `Completed: ${index + 1} / ${files.length} · ` +
                `Successful: ${succeeded} · Failed: ${failed}`;
        }

        if (lastSuccessfulData) {
            selectedImage = lastSuccessfulFile;
            selectedImages = [lastSuccessfulFile];

            selectedFile.textContent =
                `Last successfully processed: ${lastSuccessfulFile.name}`;

            predictButton.textContent = originalPredictButtonText;

            await displayPredictionResult(lastSuccessfulData, lastSuccessfulFile);
        }

        batchProgress.textContent =
            `Batch processing finished. ${succeeded} successful, ${failed} failed.`;
    }

    predictButton.addEventListener("click", async function () {
        const filesToProcess = selectedImages.length
            ? selectedImages.slice()
            : (selectedImage ? [selectedImage] : []);

        if (!filesToProcess.length) {
            showError("Please select an image first.");
            return;
        }

        predictButton.disabled = true;
        loading.style.display = "flex";
        errorBox.style.display = "none";

        try {
            if (filesToProcess.length > 1) {
                await runBatchSegmentation(filesToProcess);
            } else {
                const data = await window.SegmentationModel.predict(
                    filesToProcess[0],
                    API_URL
                );

                dispatchSegmentationHistory(data);
                await displayPredictionResult(data, filesToProcess[0]);
            }
        } catch (error) {
            console.error("Segmentation failed:", error);
            showError(error.message || String(error));
        } finally {
            predictButton.disabled = false;
            loading.style.display = "none";
        }
    });

        function resetTraversabilityStatistics() {
            const ids = [
                "travHighPercentage", "travHighPixels",
                "travMediumPercentage", "travMediumPixels",
                "travLowPercentage", "travLowPixels",
                "travBlockedPercentage", "travBlockedPixels"
            ];

            ids.forEach(function (id) {
                const element = document.getElementById(id);
                if (element) element.textContent = "—";
            });
        }

        function renderTraversabilityStatistics(statistics) {
            const rows = [
                ["high", "travHighPercentage", "travHighPixels"],
                ["medium", "travMediumPercentage", "travMediumPixels"],
                ["low", "travLowPercentage", "travLowPixels"],
                ["blocked", "travBlockedPercentage", "travBlockedPixels"]
            ];

            rows.forEach(function ([key, percentageId, pixelsId]) {
                const item = statistics[key];

                if (!item) return;

                document.getElementById(percentageId).textContent =
                    `${item.percentage}%`;

                document.getElementById(pixelsId).textContent =
                    `${Number(item.pixels).toLocaleString()} px`;
            });
        }

        function updateConfidenceLegend(low, high) {
            confidenceLowThresholdValue.textContent = `${low}%`;
            confidenceHighThresholdValue.textContent = `${high}%`;

            confidenceLowLegend.textContent =
                `Low confidence — below ${low}%`;

            confidenceMediumLegend.textContent =
                `Medium confidence — ${low}–${high}%`;

            confidenceHighLegend.textContent =
                `High confidence — ${high}% and above`;
        }

        function renderConfidenceStatistics(result) {
            const statistics = result.statistics || {};

            function formatCategory(name) {
                const item = statistics[name];

                if (!item) return "—";

                return `${item.percentage}% (${Number(item.pixels).toLocaleString()} px)`;
            }

            meanConfidence.textContent =
                `${result.mean_confidence}%`;

            lowConfidence.textContent = formatCategory("low");
            mediumConfidence.textContent = formatCategory("medium");
            highConfidence.textContent = formatCategory("high");
        }

        async function recalculateConfidence(silent = false, displayId = displayRequestId) {
            const requestedFilename = confidenceFilename;
            if (!confidenceFilename) {
                confidenceStatus.textContent =
                    "Cannot recalculate: confidence data file is unavailable.";
                return;
            }

            const low = Number(confidenceLowThresholdInput.value);
            const high = Number(confidenceHighThresholdInput.value);

            updateConfidenceLegend(low, high);

            if (low >= high) {
                confidenceStatus.textContent =
                    "Low threshold must be lower than High threshold.";
                return;
            }

            recalculateConfidenceButton.disabled = true;
            recalculateConfidenceButton.textContent = "Recalculating...";

            if (!silent) {
                confidenceStatus.textContent =
                    "Recalculating confidence map...";
            }

            try {
                const response = await fetch(
                    `${API_URL}/confidence/recalculate`,
                    {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({
                            confidence_filename: requestedFilename,
                            low_threshold: low,
                            high_threshold: high
                        })
                    }
                );

                const result = await response.json();

                if (displayId !== displayRequestId) return;

                if (!response.ok) {
                    throw new Error(
                        result.detail || "Confidence map recalculation failed."
                    );
                }

                confidenceUrl =
                    `${API_URL}${result.confidence_url}`;

                window.Auth.invalidateProtectedImage(confidenceUrl);

                downloadConfidence.href =
                    `${confidenceUrl}?download=true`;

                downloadConfidence.download =
                    result.confidence_url.split("/").pop();

                renderConfidenceStatistics(result);

                confidenceStatus.textContent =
                    `Map updated successfully. Low: below ${low}%, ` +
                    `Medium: ${low}–${high}%, High: ${high}% and above.`;

                updateComparisonOptions();

                if (confidenceButton.classList.contains("active")) {
                    setResultView("confidence");
                }

                if (comparisonSelect2.value === "confidence") {
                    renderComparisonPanel2("confidence");
                }

            } catch (error) {
                if (displayId !== displayRequestId) return;

                console.error(
                    "Confidence recalculation failed:",
                    error
                );

                confidenceStatus.textContent =
                    `Recalculation error: ${error.message}`;

                if (!silent) {
                    showError(error.message);
                }

            } finally {
                recalculateConfidenceButton.disabled = false;
                recalculateConfidenceButton.textContent =
                    "Recalculate Map";
            }
        }

        confidenceLowThresholdInput.addEventListener("input", function () {
            const low = Number(this.value);
            let high = Number(confidenceHighThresholdInput.value);

            if (low >= high) {
                high = Math.min(100, low + 1);
                confidenceHighThresholdInput.value = high;
            }

            updateConfidenceLegend(low, high);
        });

        confidenceHighThresholdInput.addEventListener("input", function () {
            const high = Number(this.value);
            let low = Number(confidenceLowThresholdInput.value);

            if (high <= low) {
                low = Math.max(0, high - 1);
                confidenceLowThresholdInput.value = low;
            }

            updateConfidenceLegend(low, high);
        });

        recalculateConfidenceButton.addEventListener("click", function () {
            recalculateConfidence(false);
        });

        async function recalculateTraversability(silent = false, displayId = displayRequestId) {
            const requestedClassIdsFilename = classIdsFilename;
            if (!classIdsFilename) {
                traversabilityStatus.textContent =
                    "Cannot recalculate: class-ID mask is unavailable.";
                return;
            }

            const radius = Number(obstacleRadiusInput.value);

            recalculateTraversabilityButton.disabled = true;
            recalculateTraversabilityButton.textContent = "Recalculating...";

            if (!silent) {
                traversabilityStatus.textContent =
                    "Recalculating traversability map...";
            }

            try {
                const response = await fetch(
                    `${API_URL}/traversability/recalculate`,
                    {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify({
                            class_ids_filename: requestedClassIdsFilename,
                            obstacle_radius: radius
                        })
                    }
                );

                const result = await response.json();

                if (displayId !== displayRequestId) return;

                if (!response.ok) {
                    throw new Error(
                        result.detail || "Traversability recalculation failed."
                    );
                }

                traversabilityUrl =
                    `${API_URL}${result.traversability_url}`;

                window.Auth.invalidateProtectedImage(traversabilityUrl);

                downloadTraversability.href =
                    `${traversabilityUrl}${traversabilityUrl.includes("?") ? "&" : "?"}download=true`;
                downloadTraversability.download =
                    result.traversability_url.split("/").pop();

                renderTraversabilityStatistics(result.statistics);

// Update Safe Corridor using the recalculated traversability map.
                if (result.safe_corridor_url) {
                    safeCorridorUrl = `${API_URL}${result.safe_corridor_url}`;
                    window.Auth.invalidateProtectedImage(safeCorridorUrl);

                    safeCorridorButton.disabled = false;
                    downloadSafeCorridor.style.display = "inline-flex";

                    downloadSafeCorridor.href =
                        `${safeCorridorUrl}${safeCorridorUrl.includes("?") ? "&" : "?"}download=true`;

                    downloadSafeCorridor.download =
                        result.safe_corridor_url.split("/").pop();

                    renderSafeCorridorStatistics(result.safe_corridor_statistics);

                    if (safeCorridorButton.classList.contains("active")) {
                        setResultView("safe_corridor");
                    }
                } else {
                    safeCorridorUrl = "";
                    safeCorridorButton.disabled = true;
                    downloadSafeCorridor.style.display = "none";
                    safeCorridorLegend.style.display = "none";
                    safeCorridorMetrics.style.display = "none";
                }

                traversabilityStatus.textContent =
                    `Map updated successfully. Obstacle radius: ${result.obstacle_radius} px.`;

                if (traversabilityButton.classList.contains("active")) {
                    setResultView("traversability");
                }

            } catch (error) {
                if (displayId !== displayRequestId) return;

                console.error("Traversability recalculation failed:", error);

                traversabilityStatus.textContent =
                    `Recalculation error: ${error.message}`;

                if (!silent) {
                    showError(error.message);
                }

            } finally {
                recalculateTraversabilityButton.disabled = false;
                recalculateTraversabilityButton.textContent =
                    "Recalculate Map";
            }
        }

        obstacleRadiusInput.addEventListener("input", function () {
            obstacleRadiusValue.textContent = this.value;
        });

        recalculateTraversabilityButton.addEventListener("click", function () {
            recalculateTraversability(false);
        });


        /* =====================
        IMAGE MODE CONTROLS
        ======================== */

        function getComparisonModes() {
            return {
                overlay: {
                    url: overlayUrl,
                    title: "Segmentation overlay"
                },
                mask: {
                    url: maskUrl,
                    title: "Segmentation mask"
                },
                traversability: {
                    url: traversabilityUrl,
                    title: "Traversability Map"
                },
                confidence: {
                    url: confidenceUrl,
                    title: "Model Confidence Map"
                },
                risk: {
                    url: riskMapUrl,
                    title: "Risk Map"
                },
                safe_corridor: {
                    url: safeCorridorUrl,
                    title: "Safe Corridor"
                },
                depth: {
                    url: depthUrl,
                    title: "Relative Depth Map"
                },
                uncertainty: {
                    url: uncertaintyUrl,
                    title: "Uncertainty Map"
                },
            };
        }

        function updateComparisonOptions() {
            [comparisonSelect1, comparisonSelect2].forEach(select => {
                if (!select) return;

                [
                    ["depth", "Depth Map"],
                    ["uncertainty", "Uncertainty Map"]
                ].forEach(([value, label]) => {
                    if (!select.querySelector(`option[value="${value}"]`)) {
                        const option = document.createElement("option");
                        option.value = value;
                        option.textContent = label;
                        select.appendChild(option);
                    }
                });
            });

            const modes = getComparisonModes();

            [comparisonSelect1, comparisonSelect2].forEach(select => {
                Object.entries(modes).forEach(([key, mode]) => {
                    const option = select.querySelector(
                        `option[value="${key}"]`
                    );

                    if (option) {
                        option.disabled = !mode.url;
                    }
                });
            });
        }

        function renderComparisonPanel2(mode) {
            const selectedMode = getComparisonModes()[mode];

            if (!selectedMode || !selectedMode.url) {
                return;
            }

            window.SegmentationView.showImage(
                comparisonImage2,
                selectedMode.url
            );

            comparisonTitle2.textContent = selectedMode.title;
        }

        function setResultView(mode) {
            const modes = {
                overlay: {
                    url: overlayUrl,
                    title: "Segmentation overlay"
                },
                mask: {
                    url: maskUrl,
                    title: "Segmentation mask"
                },
                traversability: {
                    url: traversabilityUrl,
                    title: "Traversability Map"
                },
                confidence: {
                    url: confidenceUrl,
                    title: "Model Confidence Map"
                },
                risk: {
                    url: riskMapUrl,
                    title: "Risk Map"
                },
                safe_corridor: {
                    url: safeCorridorUrl,
                    title: "Safe Corridor"
                },
                depth: {
                    url: depthUrl,
                    title: "Relative Depth Map"
                },
                uncertainty: {
                    url: uncertaintyUrl,
                    title: "Uncertainty Map"
                },
            };

            const selectedMode = modes[mode];

            if (!selectedMode || !selectedMode.url) {
                return;
            }

            // Display selected image
            window.SegmentationView.showImage(
                resultImage,
                selectedMode.url
            );

            resultTitle.textContent = selectedMode.title;
            comparisonSelect1.value = mode;

            // Only one button can be active
            overlayButton.classList.toggle("active", mode === "overlay");
            maskButton.classList.toggle("active", mode === "mask");
            traversabilityButton.classList.toggle(
                "active",
                mode === "traversability"
            );
            confidenceButton.classList.toggle("active", mode === "confidence");
            riskButton.classList.toggle("active", mode === "risk");
            safeCorridorButton.classList.toggle(
                "active",
                mode === "safe_corridor"
            );

            depthButton.classList.toggle("active", mode === "depth");
            uncertaintyButton.classList.toggle(
                "active",
                mode === "uncertainty"
            );

            riskLegend.style.display =
                mode === "risk" && riskMapUrl ? "block" : "none";

            riskMetrics.style.display =
                mode === "risk" && riskMapUrl ? "block" : "none";

            safeCorridorLegend.style.display =
                mode === "safe_corridor" && safeCorridorUrl ? "block" : "none";

            safeCorridorMetrics.style.display =
                mode === "safe_corridor" && safeCorridorUrl ? "block" : "none";

            // Show only the information relevant to the selected view.
            traversabilityLegend.style.display =
                mode === "traversability" && traversabilityUrl ? "block" : "none";

            confidenceLegend.style.display =
                mode === "confidence" && confidenceUrl ? "block" : "none";

            confidenceMetrics.style.display =
                mode === "confidence" && confidenceUrl ? "block" : "none";
        }

        depthButton.addEventListener("click", function () {
            setResultView("depth");
        });

        uncertaintyButton.addEventListener("click", function () {
            setResultView("uncertainty");
        });

        overlayButton.addEventListener("click", function () {
            setResultView("overlay");
        });

        maskButton.addEventListener("click", function () {
            setResultView("mask");
        });

        traversabilityButton.addEventListener("click", function () {
            setResultView("traversability");
        });

        confidenceButton.addEventListener("click", function () {
            setResultView("confidence");
        });

        riskButton.addEventListener("click", function () {
            setResultView("risk");
        });

        safeCorridorButton.addEventListener("click", function () {
            setResultView("safe_corridor");
        });






        comparisonSelect1.addEventListener("change", function () {
            setResultView(this.value);
        });

        comparisonSelect2.addEventListener("change", function () {
            renderComparisonPanel2(this.value);
        });



    /* =======
       ERROR
    ======= */

    function showError(message) {

        errorBox.textContent =
            message;


        errorBox.style.display =
            "block";

    }
