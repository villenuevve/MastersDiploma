(() => {
    "use strict";

    const API_URL = "http://127.0.0.1:8000";

    const datasetPanel = document.getElementById("datasetPanel");
    const adminPanel = document.getElementById("adminPanel");

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
    const groundTruthFilter = document.getElementById("groundTruthFilter");
    const createDatasetButton = document.getElementById("createDatasetButton");
    const editDatasetButton = document.getElementById("editDatasetButton");
    const createSplitsButton = document.getElementById("createSplitsButton");
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

        if (!canManageDatasets) {
            datasets = [];
            datasetSelect.innerHTML = '<option value="">Select dataset</option>';
            datasetImagesBody.innerHTML =
                '<tr><td colspan="8">Select a dataset to view images.</td></tr>';
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
                    '<tr><td colspan="8">No images in this dataset.</td></tr>';
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

    function selectedDataset() {
        return datasets.find(item => item.id === Number(datasetSelect.value)) || null;
    }

    function isSystemDataset(dataset) {
        return Boolean(dataset && dataset.name === "Uploaded images");
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

    async function loadDatasetImages(datasetId) {
        if (!datasetId) return;

        datasetImagesBody.innerHTML =
            '<tr><td colspan="8">Loading images...</td></tr>';

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
            createSplitsButton.disabled = systemDataset || splits.length >= 3;
            deleteDatasetButton.hidden =
                !currentUser || currentUser.role !== "Administrator";
            deleteDatasetButton.disabled =
                systemDataset || Number(data.image_count) > 0 || splits.length > 0;

            if (systemDataset) {
                datasetSummary.textContent += " · System dataset";
            }

            if (!Array.isArray(data.images) || data.images.length === 0) {
                datasetImagesBody.innerHTML =
                    '<tr><td colspan="8">No images in this dataset.</td></tr>';
                return;
            }

            const searchTerm = datasetImageSearch.value.trim().toLocaleLowerCase();
            const gtFilter = groundTruthFilter.value;

            const visibleImages = data.images.filter(image => {
                const matchesName = String(image.filename || "")
                    .toLocaleLowerCase()
                    .includes(searchTerm);

                const hasGroundTruth = Boolean(image.has_ground_truth);

                const matchesGroundTruth =
                    gtFilter === "all" ||
                    (gtFilter === "with" && hasGroundTruth) ||
                    (gtFilter === "without" && !hasGroundTruth);

                return matchesName && matchesGroundTruth;
            });

            datasetSummary.textContent =
                `${data.dataset_name} · ${data.image_count} images · Showing ${visibleImages.length}`;

            if (visibleImages.length === 0) {
                datasetImagesBody.innerHTML =
                    '<tr><td colspan="8">No images match the selected filters.</td></tr>';
                return;
            }

            const movableDatasets = datasets.filter(item =>
                item.id !== Number(datasetId) && !isSystemDataset(item)
            );

            datasetImagesBody.innerHTML = visibleImages.map(image => {
                const splitOptions = [
                    '<option value="">Unassigned</option>',
                    ...splits.map(split => `
                        <option value="${split.id}"
                            ${image.split_id === split.id ? "selected" : ""}>
                            ${escapeHtml(split.name)}
                        </option>
                    `)
                ].join("");

                const moveOptions = [
                    '<option value="">Move to...</option>',
                    ...movableDatasets.map(item => `
                        <option value="${item.id}">
                            ${escapeHtml(item.name)}
                        </option>
                    `)
                ].join("");

                const hasGroundTruth = Boolean(image.has_ground_truth);

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
                            <div class="ground-truth-cell">
                                <span class="mask-status ${hasGroundTruth ? "is-active" : "is-inactive"}">
                                    ${hasGroundTruth ? "Yes" : "No"}
                                </span>
                                <input
                                    class="ground-truth-file"
                                    type="file"
                                    accept="image/png"
                                    data-image-id="${image.id}"
                                    aria-label="Upload Ground Truth for ${escapeHtml(image.filename)}"
                                    ${systemDataset ? "disabled" : ""}
                                >
                                ${hasGroundTruth ? `
                                    <button
                                        class="ground-truth-delete-button danger-button"
                                        type="button"
                                        data-image-id="${image.id}"
                                        ${systemDataset ? "disabled" : ""}
                                    >Remove GT</button>
                                ` : ""}
                            </div>
                        </td>
                        <td>
                            <select class="image-split-select"
                                data-image-id="${image.id}"
                                ${systemDataset || splits.length === 0 ? "disabled" : ""}>
                                ${splitOptions}
                            </select>
                        </td>
                        <td>
                            <select class="image-move-select"
                                data-image-id="${image.id}">
                                ${moveOptions}
                            </select>
                        </td>
                        <td>
                            <button class="image-delete-button danger-button"
                                type="button"
                                data-image-id="${image.id}"
                                ${systemDataset ? "disabled" : ""}>
                                Delete
                            </button>
                        </td>
                    </tr>
                `;
            }).join("");
        } catch (error) {
            datasetImagesBody.innerHTML =
                '<tr><td colspan="8">Failed to load data.</td></tr>';
            showMessage(datasetMessage, error.message, true);
        }
    }

    async function uploadGroundTruth(imageId, file) {
        const dataset = selectedDataset();

        if (!dataset || isSystemDataset(dataset) || !file) return;

        if (
            file.type !== "image/png" ||
            !file.name.toLowerCase().endsWith(".png")
        ) {
            showMessage(datasetMessage, "Ground Truth must be a PNG file.", true);
            return;
        }

        if (file.size > 20 * 1024 * 1024) {
            showMessage(datasetMessage, "Ground Truth file exceeds 20 MB.", true);
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            const result = await request(
                `/datasets/${dataset.id}/images/${imageId}/ground-truth`,
                {
                    method: "POST",
                    body: formData
                }
            );

            showMessage(
                datasetMessage,
                `Ground Truth uploaded. Classes: ${result.classes.length}.`
            );

            await loadDatasetImages(dataset.id);
        } catch (error) {
            showMessage(datasetMessage, error.message, true);
        }
    }

    async function deleteGroundTruth(imageId) {
        const dataset = selectedDataset();

        if (!dataset || isSystemDataset(dataset)) return;

        if (!confirm("Delete the Ground Truth mask for this image?")) return;

        try {
            await request(
                `/datasets/${dataset.id}/images/${imageId}/ground-truth`,
                { method: "DELETE" }
            );

            showMessage(datasetMessage, "Ground Truth deleted.");
            await loadDatasetImages(dataset.id);
        } catch (error) {
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

        if (!confirm(
            `Create Train, Val and Test splits for "${dataset.name}"?`
        )) return;

        try {
            await request(`/datasets/${dataset.id}/splits`, {
                method: "POST"
            });

            datasetSplitFilter.value = "";
            showMessage(datasetMessage, "Train / Val / Test splits are ready.");
            await loadDatasetImages(dataset.id);
        } catch (error) {
            showMessage(datasetMessage, error.message, true);
        }
    }

    async function deleteDataset() {
        const dataset = selectedDataset();

        if (!dataset || isSystemDataset(dataset)) return;

        if (!confirm(
            `Delete empty dataset "${dataset.name}"? This action cannot be undone.`
        )) return;

        try {
            await request(`/datasets/${dataset.id}`, {
                method: "DELETE"
            });

            showMessage(datasetMessage, "Dataset deleted successfully.");
            await loadDatasets();
        } catch (error) {
            showMessage(datasetMessage, error.message, true);
        }
    }

    async function assignImageSplit(imageId, splitId) {
        const dataset = selectedDataset();
        if (!dataset || isSystemDataset(dataset)) return;

        try {
            await request(
                `/datasets/${dataset.id}/images/${imageId}/split`,
                {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        split_id: splitId ? Number(splitId) : null
                    })
                }
            );

            showMessage(datasetMessage, "Image split updated.");
            await loadDatasetImages(dataset.id);
        } catch (error) {
            showMessage(datasetMessage, error.message, true);
            await loadDatasetImages(dataset.id);
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

        const oversizedFile = files.find(file => file.size > 20 * 1024 * 1024);

        if (oversizedFile) {
            showMessage(
                datasetMessage,
                `File exceeds 20 MB: ${oversizedFile.name}`,
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
    createSplitsButton.addEventListener("click", createDatasetSplits);
    deleteDatasetButton.addEventListener("click", deleteDataset);

    datasetSplitFilter.addEventListener("change", () => {
        loadDatasetImages(Number(datasetSelect.value));
    });

    datasetImageSearch.addEventListener("input", () => {
        loadDatasetImages(Number(datasetSelect.value));
    });

    groundTruthFilter.addEventListener("change", () => {
        loadDatasetImages(Number(datasetSelect.value));
    });

    datasetImagesBody.addEventListener("change", event => {
        const groundTruthInput = event.target.closest(".ground-truth-file");

        if (groundTruthInput && !groundTruthInput.disabled) {
            const file = groundTruthInput.files[0];
            const imageId = Number(groundTruthInput.dataset.imageId);
            groundTruthInput.value = "";
            if (file) uploadGroundTruth(imageId, file);
            return;
        }

        const splitSelect = event.target.closest(".image-split-select");

        if (splitSelect && !splitSelect.disabled) {
            assignImageSplit(
                Number(splitSelect.dataset.imageId),
                splitSelect.value
            );
            return;
        }

        const moveSelect = event.target.closest(".image-move-select");

        if (moveSelect && moveSelect.value) {
            moveDatasetImage(
                Number(moveSelect.dataset.imageId),
                Number(moveSelect.value)
            );
        }
    });

    datasetImagesBody.addEventListener("click", event => {
        const groundTruthButton = event.target.closest(".ground-truth-delete-button");

        if (groundTruthButton && !groundTruthButton.disabled) {
            deleteGroundTruth(Number(groundTruthButton.dataset.imageId));
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
