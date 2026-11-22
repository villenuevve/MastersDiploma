(function () {
    "use strict";

    const API_URL = "http://127.0.0.1:8000";
    const TOKEN_KEY = "segmentation_access_token";

    const authSection = document.getElementById("authSection");
    const authForm = document.getElementById("authForm");
    const authTitle = document.getElementById("authTitle");
    const authDescription = document.getElementById("authDescription");
    const authUsername = document.getElementById("authUsername");
    const usernameField = document.getElementById("usernameField");
    const authEmail = document.getElementById("authEmail");
    const authPassword = document.getElementById("authPassword");
    const authSubmit = document.getElementById("authSubmit");
    const authMessage = document.getElementById("authMessage");
    const authModeToggle = document.getElementById("authModeToggle");
    const appMain = document.getElementById("appMain");
    const accountBar = document.getElementById("accountBar");
    const accountUsername = document.getElementById("accountUsername");
    const accountRole = document.getElementById("accountRole");
    const logoutButton = document.getElementById("logoutButton");

    let registerMode = false;
    let currentUser = null;
    const imageObjectUrls = new WeakMap();
    const imageRequestIds = new WeakMap();
    const protectedImageCache = new Map();
    const MAX_PROTECTED_IMAGE_CACHE_SIZE = 12;
    let protectedImageCacheGeneration = 0;

    function normalizeProtectedImageUrl(url) {
        const parsedUrl = new URL(url, window.location.href);
        parsedUrl.searchParams.delete("t");
        parsedUrl.hash = "";
        return parsedUrl.origin + parsedUrl.pathname + parsedUrl.search;
    }

    function clearProtectedImageCache() {
        protectedImageCacheGeneration++;

        protectedImageCache.forEach(entry => {
            URL.revokeObjectURL(entry.objectUrl);
        });
        protectedImageCache.clear();
    }

    function invalidateProtectedImage(url) {
        if (!url) return;

        const normalizedUrl = normalizeProtectedImageUrl(url);

        for (const [key, entry] of protectedImageCache.entries()) {
            if (entry.url === normalizedUrl) {
                URL.revokeObjectURL(entry.objectUrl);
                protectedImageCache.delete(key);
            }
        }
    }

    window.addEventListener("auth:state-changed", clearProtectedImageCache);

    function getToken() {
        return sessionStorage.getItem(TOKEN_KEY);
    }

    function setMessage(message, type) {
        authMessage.textContent = message || "";
        authMessage.classList.toggle("is-error", type === "error");
        authMessage.classList.toggle("is-success", type === "success");
    }

    // Перетворюємо структуровані помилки API на текст.
    function formatApiError(detail, fallback) {
        if (!detail) return fallback;

        if (Array.isArray(detail)) {
            return detail.map(item => {
                const location = Array.isArray(item.loc)
                    ? item.loc.filter(part => part !== "body").join(" → ")
                    : "";
                const message = item.msg || item.message || JSON.stringify(item);
                return location ? `${location}: ${message}` : message;
            }).join("; ");
        }

        if (typeof detail === "object") {
            return detail.message || JSON.stringify(detail);
        }

        return String(detail);
    }

    function showAuthenticatedState(user) {
        currentUser = user;
        authSection.hidden = true;
        appMain.hidden = false;
        accountBar.hidden = false;
        accountUsername.textContent = user.username;
        accountRole.textContent = user.role || "User";
        window.dispatchEvent(new CustomEvent("auth:state-changed", { detail: user }));
    }

    function showUnauthenticatedState() {
        currentUser = null;
        sessionStorage.removeItem(TOKEN_KEY);
        appMain.hidden = true;
        accountBar.hidden = true;
        authSection.hidden = false;
        authPassword.value = "";
        window.dispatchEvent(new CustomEvent("auth:state-changed", { detail: null }));
    }

    function updateMode() {
        usernameField.hidden = !registerMode;
        authUsername.required = registerMode;
        authPassword.autocomplete = registerMode
            ? "new-password"
            : "current-password";

        authTitle.textContent = registerMode ? "Реєстрація" : "Вхід до системи";
        authDescription.textContent = registerMode
            ? "Створіть обліковий запис для роботи із системою."
            : "Увійдіть, щоб продовжити роботу із системою.";
        authSubmit.textContent = registerMode ? "Зареєструватися" : "Увійти";
        authModeToggle.textContent = registerMode
            ? "Вже маєте обліковий запис? Увійти"
            : "Немає облікового запису? Зареєструватися";
        setMessage("", "");
    }

    async function apiFetch(url, options) {
        const requestOptions = options || {};
        const headers = new Headers(requestOptions.headers || {});
        const token = getToken();

        if (token) {
            headers.set("Authorization", "Bearer " + token);
        }

        const response = await window.fetch(url, {
            ...requestOptions,
            headers
        });

        return response;
    }

    // Додаємо JWT до всіх запитів застосунку, адресованих API.
    const originalFetch = window.fetch.bind(window);

    window.fetch = function (input, init) {
        const rawUrl = typeof input === "string"
            ? input
            : (input && input.url) || "";

        if (rawUrl.startsWith(API_URL)) {
            const options = init || {};
            const headers = new Headers(
                options.headers || (input instanceof Request ? input.headers : {})
            );
            const token = getToken();

            if (token) {
                headers.set("Authorization", "Bearer " + token);
            }

            return originalFetch(input, { ...options, headers }).then(async response => {
                if (response.status === 401 && token &&
                    !rawUrl.includes("/auth/login")) {
                    try {
                        const verifyHeaders = new Headers();
                        verifyHeaders.set("Authorization", "Bearer " + token);

                        const verifyResponse = await originalFetch(
                            API_URL + "/auth/me",
                            { method: "GET", headers: verifyHeaders }
                        );

                        if (
                            verifyResponse.status === 401 &&
                            getToken() === token &&
                            currentUser !== null
                        ) {
                            showUnauthenticatedState();
                            setMessage(
                                "Сесію завершено. Увійдіть знову.",
                                "error"
                            );
                        }
                    } catch (error) {
                        console.error(
                            "Не вдалося перевірити стан сесії:",
                            error
                        );
                    }
                }

                return response;
            });
        }

        return originalFetch(input, init);
    };

    // Завантажуємо захищене зображення через JWT із повторним використанням кешу.
    async function loadProtectedImage(imageElement, url) {
        if (!imageElement || !url) return;

        const requestId = (imageRequestIds.get(imageElement) || 0) + 1;
        imageRequestIds.set(imageElement, requestId);

        const token = getToken();
        const cacheGeneration = protectedImageCacheGeneration;

        try {
            const normalizedUrl = normalizeProtectedImageUrl(url);
            const cacheKey = `${token || ""}|${normalizedUrl}`;
            const cachedEntry = protectedImageCache.get(cacheKey);

            if (cachedEntry) {
                if (imageRequestIds.get(imageElement) !== requestId) return;

                protectedImageCache.delete(cacheKey);
                protectedImageCache.set(cacheKey, cachedEntry);

                imageObjectUrls.set(imageElement, cachedEntry.objectUrl);
                imageElement.src = cachedEntry.objectUrl;
                return;
            }

            const response = await apiFetch(url, { method: "GET" });

            if (!response.ok) {
                throw new Error(
                    "Не вдалося завантажити зображення: HTTP " + response.status
                );
            }

            const blob = await response.blob();

            if (imageRequestIds.get(imageElement) !== requestId) return;
            if (getToken() !== token) return;
            if (protectedImageCacheGeneration !== cacheGeneration) return;

            const objectUrl = URL.createObjectURL(blob);

            protectedImageCache.set(cacheKey, {
                url: normalizedUrl,
                objectUrl
            });

            while (protectedImageCache.size > MAX_PROTECTED_IMAGE_CACHE_SIZE) {
                const oldestKey = protectedImageCache.keys().next().value;
                const oldestEntry = protectedImageCache.get(oldestKey);

                URL.revokeObjectURL(oldestEntry.objectUrl);
                protectedImageCache.delete(oldestKey);
            }

            imageObjectUrls.set(imageElement, objectUrl);
            imageElement.src = objectUrl;

        } catch (error) {
            if (imageRequestIds.get(imageElement) !== requestId) return;
            console.error("Помилка завантаження захищеного зображення:", error);
        }
    }

    // Завантажуємо захищений файл із передаванням токена авторизації.
    async function downloadProtectedFile(url, filename) {
        const response = await apiFetch(url, { method: "GET" });

        if (!response.ok) {
            throw new Error("Помилка завантаження файлу: HTTP " + response.status);
        }

        const blob = await response.blob();
        const objectUrl = URL.createObjectURL(blob);
        const link = document.createElement("a");

        link.href = objectUrl;
        link.download = filename || "download";
        document.body.appendChild(link);
        link.click();
        link.remove();

        setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    }

    // Перехоплюємо посилання на захищені результати, зокрема в пакетних картках.
    document.addEventListener("click", function (event) {
        const link = event.target.closest("a");

        if (!link || !link.href) return;
        if (!link.href.startsWith(API_URL + "/results/")) return;

        event.preventDefault();
        event.stopPropagation();

        const filename = link.download ||
            decodeURIComponent(link.href.split("/").pop().split("?")[0]) ||
            "download";

        downloadProtectedFile(link.href, filename).catch(error => {
            console.error(error);
            setMessage(error.message || "Не вдалося завантажити файл.", "error");
        });
    }, true);

    authModeToggle.addEventListener("click", function () {
        registerMode = !registerMode;
        updateMode();
    });

    // Обробляємо реєстрацію та вхід до системи.
    authForm.addEventListener("submit", async function (event) {
        event.preventDefault();
        setMessage("", "");
        authSubmit.disabled = true;

        try {
            if (registerMode) {
                const response = await originalFetch(API_URL + "/auth/register", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        username: authUsername.value.trim(),
                        email: authEmail.value.trim(),
                        password: authPassword.value
                    })
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(formatApiError(data.detail, "Помилка реєстрації."));
                }

                registerMode = false;
                updateMode();
                authPassword.value = "";
                setMessage(
                    "Обліковий запис створено. Тепер увійдіть до системи.",
                    "success"
                );
                return;
            }

            const response = await originalFetch(API_URL + "/auth/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    email: authEmail.value.trim(),
                    password: authPassword.value
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(formatApiError(data.detail, "Неправильний email або пароль."));
            }

            sessionStorage.setItem(TOKEN_KEY, data.access_token);

            const profileResponse = await apiFetch(API_URL + "/auth/me");
            const profile = await profileResponse.json();

            if (!profileResponse.ok) {
                throw new Error(formatApiError(profile.detail, "Не вдалося отримати профіль."));
            }

            authPassword.value = "";
            showAuthenticatedState(profile);
        } catch (error) {
            setMessage(error.message || "Помилка авторизації.", "error");
        } finally {
            authSubmit.disabled = false;
        }
    });

    // Завершуємо поточну сесію користувача.
    logoutButton.addEventListener("click", function () {
        showUnauthenticatedState();
        setMessage("Ви вийшли із системи.", "success");
    });

    // Перевіряємо наявність і чинність збереженої сесії під час запуску.
    async function initialize() {
        const token = getToken();

        if (!token) {
            showUnauthenticatedState();
            document.documentElement.classList.remove("auth-pending");
            return;
        }

        try {
            const response = await apiFetch(API_URL + "/auth/me");

            if (response.status === 401) {
                showUnauthenticatedState();
                setMessage(
                    "Сесію завершено. Увійдіть знову.",
                    "error"
                );
                return;
            }

            if (!response.ok) {
                setMessage(
                    "Сервер тимчасово не може перевірити сесію. Оновіть сторінку пізніше.",
                    "error"
                );
                return;
            }

            const data = await response.json();
            showAuthenticatedState(data);

        } catch (error) {
            console.error("Не вдалося перевірити сесію:", error);
            setMessage(
                "Не вдалося зв'язатися із сервером. Сесію збережено — спробуйте оновити сторінку.",
                "error"
            );
        } finally {
            document.documentElement.classList.remove("auth-pending");
        }
    }

    window.Auth = {
        getToken,
        apiFetch,
        loadProtectedImage,
        invalidateProtectedImage,
        downloadProtectedFile,
        getCurrentUser: () => currentUser
    };

    initialize();
})();