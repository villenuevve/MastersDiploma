(() => {
    "use strict";

    const API_URL = "http://127.0.0.1:8000";

    const datasetPanel = document.getElementById("datasetPanel");
    const adminPanel = document.getElementById("adminPanel");
    const datasetsNavLink = document.getElementById("navDatasets");
    const usersNavLink = document.getElementById("navUsers");

    if (!datasetPanel || !adminPanel || !window.Auth) {
        console.error("Role panels could not be initialized.");
        return;
    }

    const datasetSelect = document.getElementById("datasetSelect");
    const datasetImagesBody = document.getElementById("datasetImagesBody");
    const datasetSummary = document.getElementById("datasetSummary");
    const datasetFiles = document.getElementById("datasetFiles");
    const datasetMessage = document.getElementById("datasetMessage");
    const datasetSplitFilter = document.getElementById("datasetSplitFilter");
    const datasetImageSearch = document.getElementById("datasetImageSearch");
        const createDatasetButton = document.getElementById("createDatasetButton");
    const editDatasetButton = document.getElementById("editDatasetButton");
    const downloadDatasetButton = document.getElementById("downloadDatasetButton");
    const downloadDatasetWithSegmentationButton =
        document.getElementById("downloadDatasetWithSegmentationButton");
    const importDatasetFile = document.getElementById("importDatasetFile");
    const importDatasetButton = document.getElementById("importDatasetButton");
    const createSplitsButton = document.getElementById("createSplitsButton");
    const trainPercentInput = document.getElementById("trainPercent");
    const valPercentInput = document.getElementById("valPercent");
    const testPercentInput = document.getElementById("testPercent");
    const splitSeedInput = document.getElementById("splitSeed");
    const deleteDatasetButton = document.getElementById("deleteDatasetButton");
    const usersTableBody = document.getElementById("usersTableBody");
    const usersMessage = document.getElementById("usersMessage");
    const toggleCreateUserButton = document.getElementById("toggleCreateUserButton");
    const adminCreateUserForm = document.getElementById("adminCreateUserForm");
    const createUserSubmitButton = document.getElementById("createUserSubmitButton");

    let currentUser = null;
    let datasets = [];

    function escapeHtml(value) {
        return String(value ?? "").replace(/[&<>"']/g, char => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;"
        })[char]);
    }

    function showMessage(element, message, isError = false) {
        element.textContent = message || "";
        element.classList.toggle("is-error", Boolean(isError));
        element.classList.toggle("is-success", Boolean(message) && !isError);
    }

    async function request(path, options = {}) {
        const response = await window.Auth.apiFetch(API_URL + path, options);
        let data = {};

        try {
            data = await response.json();
        } catch (_) {
            data = {};
        }

        if (!response.ok) {
            const detail = data.detail ?? data.message;
            let errorMessage = "Unknown error.";

            if (typeof detail === "string") {
                errorMessage = detail;
            } else if (Array.isArray(detail)) {
                errorMessage = detail.map(item => {
                    if (typeof item === "string") return item;

                    if (item && typeof item === "object") {
                        const location = Array.isArray(item.loc)
                            ? item.loc.join(" → ")
                            : "";
                        const message = item.msg || JSON.stringify(item);

                        return location
                            ? `${location}: ${message}`
                            : message;
                    }

                    return String(item);
                }).join("; ");
            } else if (detail && typeof detail === "object") {
                errorMessage = detail.msg || JSON.stringify(detail);
            }

            throw new Error(errorMessage);
        }

        return data;
    }


    function formatBytes(bytes) {
        const value = Number(bytes);

        if (!Number.isFinite(value) || value <= 0) return "—";
        if (value < 1024) return value + " B";
        if (value < 1024 * 1024) return (value / 1024).toFixed(1) + " KB";
        return (value / (1024 * 1024)).toFixed(2) + " MB";
    }

    function formatDate(value) {
        if (!value) return "—";

        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return "—";

        return date.toLocaleDateString();
    }

    function updateRolePanels(user) {
        currentUser = user || null;

        const role = currentUser ? currentUser.role : null;
        const canManageDatasets = role === "Technician" || role === "Administrator";
        const isAdmin = role === "Administrator";

        datasetPanel.hidden = !canManageDatasets;
        adminPanel.hidden = !isAdmin;

        if (datasetsNavLink) {
            datasetsNavLink.hidden = !canManageDatasets;
        }

        if (usersNavLink) {
            usersNavLink.hidden = !isAdmin;
        }

        if (!canManageDatasets) {
            datasets = [];
            datasetSelect.innerHTML = '<option value="">Select dataset</option>';
            datasetImagesBody.innerHTML =
                '<tr><td colspan="7">Select a dataset to view images.</td></tr>';
        }

        if (isAdmin) {
            loadUsers();
        }

        if (canManageDatasets) {
            loadDatasets();
        }
    }

    async function loadUsers() {
        if (!currentUser || currentUser.role !== "Administrator") return;

        showMessage(usersMessage, "");

        try {
            const users = await request("/auth/users");

            if (!Array.isArray(users) || users.length === 0) {
                usersTableBody.innerHTML =
                    '<tr><td colspan="6">No users found.</td></tr>';
                return;
            }

            usersTableBody.innerHTML = users.map(user => {
                const isSelf = user.id === currentUser.id;
                const roleOptions = ["User", "Technician", "Administrator"]
                    .map(role => `
                        <option value="${role}" ${user.role === role ? "selected" : ""}>
                            ${role}
                        </option>
                    `).join("");

                const statusText = user.is_active ? "Active" : "Inactive";
                const statusAction = user.is_active ? "Deactivate" : "Activate";

                return `
                    <tr>
                        <td>${user.id}</td>
                        <td>${escapeHtml(user.username)}</td>
                        <td>${escapeHtml(user.email)}</td>
                        <td>
                            <select class="user-role-select"
                                data-user-id="${user.id}"
                                ${isSelf ? "disabled" : ""}>
                                ${roleOptions}
                            </select>
                        </td>
                        <td>
                            <span class="user-status ${user.is_active ? "is-active" : "is-inactive"}">
                                ${statusText}
                            </span>
                            <button
                                class="user-status-button"
                                type="button"
                                data-user-id="${user.id}"
                                data-active="${user.is_active ? "true" : "false"}"
                                ${isSelf ? "disabled" : ""}>
                                ${statusAction}
                            </button>
                        </td>
                        <td>${formatDate(user.created_at)}</td>
                    </tr>
                `;
            }).join("");
        } catch (error) {
            usersTableBody.innerHTML =
                '<tr><td colspan="6">Failed to load data.</td></tr>';
            showMessage(usersMessage, error.message, true);
        }
    }

    async function createUser(event) {
        event.preventDefault();

        if (!currentUser || currentUser.role !== "Administrator") {
            showMessage(usersMessage, "Administrator access required.", true);
            return;
        }

        const username = document.getElementById("newUsername").value.trim();
        const email = document.getElementById("newUserEmail").value.trim();
        const password = document.getElementById("newUserPassword").value;
        const role = document.getElementById("newUserRole").value;

        createUserSubmitButton.disabled = true;
        createUserSubmitButton.textContent = "Creating...";

        try {
            await request("/auth/users", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, email, password, role })
            });

            adminCreateUserForm.reset();
            adminCreateUserForm.hidden = true;
            toggleCreateUserButton.textContent = "Add User";

            await loadUsers();
            showMessage(usersMessage, "User account created successfully.");
        } catch (error) {
            showMessage(usersMessage, error.message, true);
        } finally {
            createUserSubmitButton.disabled = false;
            createUserSubmitButton.textContent = "Create account";
        }
    }

    async function loadDatasets(preferredId = null) {
        if (!currentUser ||
            !["Technician", "Administrator"].includes(currentUser.role)) return;

        showMessage(datasetMessage, "");

        try {
            datasets = await request("/datasets");

            if (!Array.isArray(datasets) || datasets.length === 0) {
                datasetSelect.innerHTML = '<option value="">No datasets found</option>';
                datasetSummary.textContent = "No datasets found.";
                datasetImagesBody.innerHTML =
                    '<tr><td colspan="7">No images in this dataset.</td></tr>';
                return;
            }

            datasetSelect.innerHTML = datasets.map(dataset => `
                <option value="${dataset.id}">
                    ${escapeHtml(dataset.name)} (${dataset.image_count} images)
                </option>
            `).join("");

            const selected = datasets.find(item => item.id === Number(preferredId))
                || datasets.find(item => item.id === Number(datasetSelect.value))
                || datasets[0];

            datasetSelect.value = String(selected.id);
            await loadDatasetImages(selected.id);
        } catch (error) {
            datasetSelect.innerHTML = '<option value="">Failed to load datasets</option>';
            showMessage(datasetMessage, error.message, true);
        }
    }

    async function downloadSelectedDataset(includePredictions = false) {
        const dataset = selectedDataset();

        if (!dataset) {
            showMessage(datasetMessage, "Select a dataset first.", true);
            return;
        }

        const button = includePredictions
            ? downloadDatasetWithSegmentationButton
            : downloadDatasetButton;

        button.disabled = true;
        const originalText = button.textContent;
        button.textContent = "Preparing ZIP...";

        try {
            const safeName = String(dataset.name || "dataset")
                .replace(/[^a-z0-9_-]+/gi, "_")
                .replace(/^_+|_+$/g, "") || "dataset";

            const suffix = includePredictions
                ? "_with_segmentation"
                : "";

            const filename = `${safeName}_${dataset.id}${suffix}.zip`;

            const url =
                `${API_URL}/datasets/${dataset.id}/export` +
                `?include_predictions=${includePredictions}`;

            await window.Auth.downloadProtectedFile(url, filename);

            showMessage(
                datasetMessage,
                includePredictions
                    ? `Dataset "${dataset.name}" with saved segmentation results downloaded successfully.`
                    : `Dataset "${dataset.name}" downloaded successfully.`
            );
        } catch (error) {
            console.error("Dataset export failed:", error);
            showMessage(
                datasetMessage,
                error.message || "Failed to download dataset.",
                true
            );
        } finally {
            button.disabled = false;
            button.textContent = originalText;
        }
    }

    async function importDataset() {
        const file = importDatasetFile.files[0];

        if (!file) {
            showMessage(datasetMessage, "Select a ZIP archive first.", true);
            return;
        }

        if (!file.name.toLowerCase().endsWith(".zip")) {
            showMessage(datasetMessage, "Only ZIP archives are supported.", true);
            return;
        }

        if (file.size > 500 * 1024 * 1024) {
            showMessage(
                datasetMessage,
                "Archive exceeds the maximum allowed size of 500 MB.",
                true
            );
            return;
        }

        importDatasetButton.disabled = true;
        importDatasetButton.textContent = "Importing...";
        showMessage(datasetMessage, "Validating and importing dataset...");

        try {
            const formData = new FormData();
            formData.append("file", file);

            const result = await request("/datasets/import", {
                method: "POST",
                body: formData
            });

            importDatasetFile.value = "";

            await loadDatasets(result.dataset_id);

            showMessage(
                datasetMessage,
                `Dataset "${result.name}" imported successfully. ` +
                `${result.image_count} images, ` +
                `${result.ground_truth_count} Ground Truth masks.`
            );
        } catch (error) {
            console.error("Dataset import failed:", error);
            showMessage(
                datasetMessage,
                error.message || "Failed to import dataset.",
                true
            );
        } finally {
            importDatasetButton.disabled = false;
            importDatasetButton.textContent = "Import Dataset";
        }
    }

    function selectedDataset() {
        return datasets.find(item => item.id === Number(datasetSelect.value)) || null;
    }

    function isSystemDataset(dataset) {
        return Boolean(dataset && dataset.is_system);
    }

    async function loadDatasetSplits(datasetId) {
        const previousSplit = datasetSplitFilter.value;

        const data = await request(`/datasets/${datasetId}/splits`);
        const splits = Array.isArray(data.splits) ? data.splits : [];

        datasetSplitFilter.innerHTML =
            '<option value="">All images</option>' +
            splits.map(split => `
                <option value="${split.id}">
                    ${escapeHtml(split.name)} (${split.image_count} images)
                </option>
            `).join("");

        if (splits.some(split => String(split.id) === previousSplit)) {
            datasetSplitFilter.value = previousSplit;
        }

        return splits;
    }

    async function openDatasetImageResult(datasetId, imageId, filename) {
        try {
            showMessage(datasetMessage, "Loading saved segmentation result...");

            const data = await request(
                `/datasets/${datasetId}/images/${imageId}/result`
            );

            const originalResponse = await window.Auth.apiFetch(
                `${API_URL}/datasets/${datasetId}/images/${imageId}/original`
            );

            if (!originalResponse.ok) {
                throw new Error(
                    `Could not load original image: HTTP ${originalResponse.status}`
                );
            }

            const originalBlob = await originalResponse.blob();
            const sourceFile = new File(
                [originalBlob],
                filename || data.filename || "image",
                { type: originalBlob.type || "image/jpeg" }
            );

            if (typeof window.openSavedPredictionResult !== "function") {
                throw new Error("Saved result viewer is not available.");
            }

            await window.openSavedPredictionResult(data, sourceFile);
            showMessage(datasetMessage, "Saved segmentation result loaded.");
        } catch (error) {
            showMessage(datasetMessage, error.message, true);
        }
    }

    async function loadDatasetImages(datasetId) {
        if (!datasetId) return;

        datasetImagesBody.innerHTML =
            '<tr><td colspan="7">Loading images...</td></tr>';

        try {
            const dataset = datasets.find(item => item.id === Number(datasetId));
            const systemDataset = isSystemDataset(dataset);

            const splits = await loadDatasetSplits(datasetId);
            const selectedSplit = datasetSplitFilter.value;

            const query = selectedSplit
                ? `?split_id=${encodeURIComponent(selectedSplit)}`
                : "";

            const data = await request(`/datasets/${datasetId}/images${query}`);

            datasetSummary.textContent =
                `${data.dataset_name} · ${data.image_count} images`;

            createDatasetButton.hidden = false;
            editDatasetButton.disabled = systemDataset;
            const totalDatasetImages = Number(dataset?.image_count ?? data.image_count ?? 0);
            const splitGeneratorDisabled = systemDataset || totalDatasetImages === 0;

            createSplitsButton.disabled = splitGeneratorDisabled;
            [trainPercentInput, valPercentInput, testPercentInput, splitSeedInput].forEach(input => {
                input.disabled = splitGeneratorDisabled;
            });
            deleteDatasetButton.hidden =
                !currentUser || currentUser.role !== "Administrator";
            deleteDatasetButton.disabled = systemDataset;

            if (systemDataset) {
                datasetSummary.textContent += " · System dataset";
            }

            if (!Array.isArray(data.images) || data.images.length === 0) {
                datasetImagesBody.innerHTML =
                    '<tr><td colspan="7">No images in this dataset.</td></tr>';
                return;
            }

            const searchTerm = datasetImageSearch.value.trim().toLocaleLowerCase();

            const visibleImages = data.images.filter(image =>
                String(image.filename || "").toLocaleLowerCase().includes(searchTerm)
            );

            datasetSummary.textContent =
                `${data.dataset_name} · ${data.image_count} images · Showing ${visibleImages.length}`;

            if (visibleImages.length === 0) {
                datasetImagesBody.innerHTML =
                    '<tr><td colspan="7">No images match the selected filters.</td></tr>';
                return;
            }

            const movableDatasets = datasets.filter(item =>
                item.id !== Number(datasetId) && !isSystemDataset(item)
            );

            datasetImagesBody.innerHTML = visibleImages.map(image => {
                const moveOptions = [
                    '<option value="">Move to...</option>',
                    ...movableDatasets.map(item => `
                        <option value="${item.id}">
                            ${escapeHtml(item.name)}
                        </option>
                    `)
                ].join("");

                return `
                    <tr>
                        <td>${escapeHtml(image.filename)}</td>
                        <td>${image.width && image.height
                            ? `${image.width} × ${image.height}` : "—"}</td>
                        <td>${formatBytes(image.file_size)}</td>
                        <td>
                            <span class="mask-status ${image.has_mask ? "is-active" : "is-inactive"}">
                                ${image.has_mask ? "Yes" : "No"}
                            </span>
                        </td>
                        <td>
                            <span data-split-name="${escapeHtml(image.split_name || "")}">
                                ${escapeHtml(image.split_name || "Not assigned")}
                            </span>
                        </td>
                        <td>
                            <select class="image-move-select"
                                data-image-id="${image.id}">
                                ${moveOptions}
                            </select>
                        </td>
                        <td class="image-actions-cell">
                            <div class="image-actions">
                                ${image.has_mask ? `
                                    <button class="image-result-button"
                                        type="button"
                                        data-image-id="${image.id}"
                                        data-dataset-id="${datasetId}">
                                        View results
                                    </button>
                                ` : ""}
                                <button class="image-delete-button danger-button"
                                    type="button"
                                    data-image-id="${image.id}"
                                    ${systemDataset ? "disabled" : ""}>
                                    Delete
                                </button>
                            </div>
                        </td>
                    </tr>
                `;
            }).join("");
        } catch (error) {
            datasetImagesBody.innerHTML =
                '<tr><td colspan="7">Failed to load data.</td></tr>';
            showMessage(datasetMessage, error.message, true);
        }
    }

    async function createDataset() {
        const name = prompt("Dataset name:");
        if (name === null) return;

        const description = prompt("Dataset description (optional):");
        if (description === null) return;

        try {
            const created = await request("/datasets", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    name: name.trim(),
                    description: description.trim() || null
                })
            });

            showMessage(datasetMessage, "Dataset created successfully.");
            await loadDatasets(created.id);
        } catch (error) {
            showMessage(datasetMessage, error.message, true);
        }
    }

    async function editDataset() {
        const dataset = selectedDataset();

        if (!dataset || isSystemDataset(dataset)) return;

        const name = prompt("Dataset name:", dataset.name);
        if (name === null) return;

        const description = prompt(
            "Dataset description:",
            dataset.description || ""
        );
        if (description === null) return;

        try {
            await request(`/datasets/${dataset.id}`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    name: name.trim(),
                    description: description.trim() || null
                })
            });

            showMessage(datasetMessage, "Dataset updated successfully.");
            await loadDatasets(dataset.id);
        } catch (error) {
            showMessage(datasetMessage, error.message, true);
        }
    }

    async function createDatasetSplits() {
        const dataset = selectedDataset();
        if (!dataset || isSystemDataset(dataset)) return;

        const train = Number(trainPercentInput.value);
        const validation = Number(valPercentInput.value);
        const test = Number(testPercentInput.value);
        const seed = Number(splitSeedInput.value);

        const values = [train, validation, test, seed];

        if (values.some(value => !Number.isInteger(value) || !Number.isFinite(value))) {
            showMessage(datasetMessage, "All values must be whole numbers.", true);
            return;
        }

        if ([train, validation, test].some(value => value < 0 || value > 100)
            || train + validation + test !== 100) {
            showMessage(
                datasetMessage,
                "Train, Validation and Test percentages must sum to 100.",
                true
            );
            return;
        }

        if (seed < 0 || seed > 4294967295) {
            showMessage(datasetMessage, "Invalid random seed.", true);
            return;
        }

        if (!confirm(
            `Regenerate split for "${dataset.name}"? All current assignments will be replaced.`
        )) return;

        createSplitsButton.disabled = true;

        try {
            const result = await request(`/datasets/${dataset.id}/splits`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    train_percent: train,
                    val_percent: validation,
                    test_percent: test,
                    seed: seed
                })
            });

            datasetSplitFilter.value = "";

            const counts = result.splits
                .map(split => `${split.name}: ${split.image_count}`)
                .join(" · ");

            showMessage(
                datasetMessage,
                `Generated ${result.total_images} images · ${counts} · Seed: ${result.seed}`
            );

            await loadDatasetImages(dataset.id);

        } catch (error) {
            showMessage(datasetMessage, error.message, true);
        } finally {
            createSplitsButton.disabled = false;
        }
    }

    async function deleteDataset() {
        const dataset = selectedDataset();

        if (!dataset || isSystemDataset(dataset)) return;

        const confirmed = confirm(
            `Permanently delete dataset "${dataset.name}"?\n\n` +
            `This will permanently delete all images, Ground Truth annotations, ` +
            `predictions, detected classes, dataset splits and associated result files.\n\n` +
            `This action cannot be undone.`
        );

        if (!confirmed) return;

        try {
            const result = await request(`/datasets/${dataset.id}`, {
                method: "DELETE"
            });

            showMessage(
                datasetMessage,
                `Dataset deleted. Images: ${result.deleted_images}, ` +
                `predictions: ${result.deleted_predictions}, ` +
                `files: ${result.deleted_files}. ` +
                `Cleanup failures: ${result.cleanup_failures}.`
            );

            await loadDatasets();

        } catch (error) {
            showMessage(datasetMessage, error.message, true);
        }
    }

    async function moveDatasetImage(imageId, targetDatasetId) {
        if (!targetDatasetId) return;

        const sourceDataset = selectedDataset();
        if (!sourceDataset) return;

        const targetDataset = datasets.find(
            item => item.id === Number(targetDatasetId)
        );

        if (!targetDataset) return;

        if (!confirm(
            `Move image to "${targetDataset.name}"?`
        )) {
            await loadDatasetImages(sourceDataset.id);
            return;
        }

        try {
            await request(
                `/images/${imageId}/dataset?target_dataset_id=${encodeURIComponent(targetDataset.id)}`,
                { method: "PATCH" }
            );

            showMessage(datasetMessage, "Image moved successfully.");
            await loadDatasets(sourceDataset.id);
        } catch (error) {
            showMessage(datasetMessage, error.message, true);
            await loadDatasetImages(sourceDataset.id);
        }
    }

    async function deleteDatasetImage(imageId) {
        const dataset = selectedDataset();

        if (!dataset || isSystemDataset(dataset)) return;

        if (!confirm(
            "Delete this image, its predictions and associated result files?"
        )) return;

        try {
            await request(
                `/datasets/${dataset.id}/images/${imageId}`,
                { method: "DELETE" }
            );

            showMessage(datasetMessage, "Image and associated files deleted.");
            await loadDatasets(dataset.id);
        } catch (error) {
            showMessage(datasetMessage, error.message, true);
        }
    }

    async function updateUserRole(userId, role) {
        try {
            await request(`/auth/users/${userId}/role`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ role })
            });

            showMessage(usersMessage, "Role updated successfully.");
            await loadUsers();
        } catch (error) {
            showMessage(usersMessage, error.message, true);
            await loadUsers();
        }
    }

    async function updateUserStatus(userId, isActive) {
        try {
            await request(`/auth/users/${userId}/status`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ is_active: isActive })
            });

            showMessage(usersMessage, "Account status updated successfully.");
            await loadUsers();
        } catch (error) {
            showMessage(usersMessage, error.message, true);
            await loadUsers();
        }
    }

    async function uploadDatasetImages() {
        const datasetId = datasetSelect.value;
        const files = Array.from(datasetFiles.files || []);

        if (!datasetId) {
            showMessage(datasetMessage, "Please select a dataset first.", true);
            return;
        }

        if (files.length === 0) {
            showMessage(datasetMessage, "Please select at least one image.", true);
            return;
        }

        const invalidFile = files.find(file =>
            !["image/jpeg", "image/png"].includes(file.type)
        );

        if (invalidFile) {
            showMessage(
                datasetMessage,
                `Unsupported file format: ${invalidFile.name}`,
                true
            );
            return;
        }

        const oversizedFile = files.find(file => file.size > 50 * 1024 * 1024);

        if (oversizedFile) {
            showMessage(
                datasetMessage,
                `File exceeds 50 MB: ${oversizedFile.name}`,
                true
            );
            return;
        }

        const button = document.getElementById("uploadDatasetButton");
        button.disabled = true;
        button.textContent = "Processing...";
        showMessage(datasetMessage, "Processing...");

        try {
            const formData = new FormData();
            files.forEach(file => formData.append("files", file));

            const result = await request(
                `/datasets/${datasetId}/images/batch`,
                { method: "POST", body: formData }
            );

            const successful = Number(result.successful || 0);
            const failed = Number(result.failed || 0);

            showMessage(
                datasetMessage,
                `Upload completed. Uploaded: ${successful}. Failed: ${failed}.`,
                failed > 0
            );

            datasetFiles.value = "";
            await loadDatasets(datasetId);
        } catch (error) {
            showMessage(datasetMessage, error.message, true);
        } finally {
            button.disabled = false;
            button.textContent = "Upload and process images";
        }
    }

    toggleCreateUserButton.addEventListener("click", () => {
        adminCreateUserForm.hidden = !adminCreateUserForm.hidden;
        toggleCreateUserButton.textContent =
            adminCreateUserForm.hidden ? "Add User" : "Cancel";
        showMessage(usersMessage, "");
    });

    adminCreateUserForm.addEventListener("submit", createUser);

    document.getElementById("refreshUsersButton")
        .addEventListener("click", loadUsers);

    document.getElementById("refreshDatasetsButton")
        .addEventListener("click", () => loadDatasets(datasetSelect.value));

    datasetSelect.addEventListener("change", () => {
        datasetSplitFilter.value = "";
        loadDatasetImages(Number(datasetSelect.value));
    });

    document.getElementById("uploadDatasetButton")
        .addEventListener("click", uploadDatasetImages);

    createDatasetButton.addEventListener("click", createDataset);
    editDatasetButton.addEventListener("click", editDataset);
    downloadDatasetButton.addEventListener(
        "click",
        () => downloadSelectedDataset(false)
    );

    downloadDatasetWithSegmentationButton.addEventListener(
        "click",
        () => downloadSelectedDataset(true)
    );
    importDatasetButton.addEventListener("click", importDataset);
    createSplitsButton.addEventListener("click", createDatasetSplits);
    deleteDatasetButton.addEventListener("click", deleteDataset);

    datasetSplitFilter.addEventListener("change", () => {
        loadDatasetImages(Number(datasetSelect.value));
    });

    datasetImageSearch.addEventListener("input", () => {
        loadDatasetImages(Number(datasetSelect.value));
    });

    datasetImagesBody.addEventListener("change", event => {
        const moveSelect = event.target.closest(".image-move-select");

        if (moveSelect && moveSelect.value) {
            moveDatasetImage(
                Number(moveSelect.dataset.imageId),
                Number(moveSelect.value)
            );
        }
    });

    datasetImagesBody.addEventListener("click", event => {
        const resultButton = event.target.closest(".image-result-button");

        if (resultButton && !resultButton.disabled) {
            resultButton.disabled = true;
            resultButton.textContent =
                document.documentElement.lang === "uk"
                    ? "Завантаження..."
                    : "Loading...";

            const row = resultButton.closest("tr");
            const filename = row?.querySelector("td")?.textContent?.trim();

            openDatasetImageResult(
                Number(resultButton.dataset.datasetId),
                Number(resultButton.dataset.imageId),
                filename
            ).finally(() => {
                resultButton.disabled = false;
                resultButton.textContent =
                    document.documentElement.lang === "uk"
                        ? "Переглянути результати"
                        : "View results";
            });

            return;
        }

        const button = event.target.closest(".image-delete-button");
        if (button && !button.disabled) {
            deleteDatasetImage(Number(button.dataset.imageId));
        }
    });

    usersTableBody.addEventListener("change", event => {
        const select = event.target.closest(".user-role-select");
        if (!select || select.disabled) return;

        updateUserRole(Number(select.dataset.userId), select.value);
    });

    usersTableBody.addEventListener("click", event => {
        const button = event.target.closest(".user-status-button");
        if (!button || button.disabled) return;

        const userId = Number(button.dataset.userId);
        const currentlyActive = button.dataset.active === "true";

        updateUserStatus(userId, !currentlyActive);
    });

    window.addEventListener("auth:state-changed", event => {
        updateRolePanels(event.detail);
    });

    const existingUser = window.Auth.getCurrentUser();
    if (existingUser) {
        updateRolePanels(existingUser);
    }
})();
