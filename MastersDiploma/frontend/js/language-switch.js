(() => {
    const uk = {
        "Semantic Segmentation Robot": "Система семантичної сегментації",
        "Computer Vision System": "Система комп'ютерного зору",
        "API connected": "API підключено",
        "Upload Image": "Завантажити зображення",
        "Choose File": "Обрати файл",
        "Segmentation Results": "Результати сегментації",
        "Original Image": "Оригінальне зображення",
        "Segmentation Mask": "Маска сегментації",
        "Overlay": "Накладання маски",
        "Download Mask": "Завантажити маску",
        "Download Overlay": "Завантажити накладання",
        "Image Information": "Інформація про зображення",
        "Filename": "Назва файлу",
        "Resolution": "Роздільна здатність",
        "Tensor Shape": "Розмірність тензора",
        "Mask Resolution": "Роздільна здатність маски",
        "Inference Time": "Час обробки",
        "Detected Classes": "Виявлені класи",
        "Model Evaluation": "Оцінювання моделі",
        "Analytics": "Аналітика",
        "History": "Історія",
        "Analysis History": "Історія аналізів",
        "The 10 most recent segmentation analyses saved in this browser.": "Останні 10 аналізів сегментації, збережених у цьому браузері.",
        "Loading history...": "Завантаження історії...",
        "Loading...": "Завантаження...",
        "Error": "Помилка",
        "Success": "Успішно",
        "Segmentation completed successfully": "Сегментацію успішно завершено",
        "road": "Дорога",
        "sidewalk": "Тротуар",
        "building": "Будівля",
        "wall": "Стіна",
        "fence": "Паркан",
        "pole": "Стовп",
        "traffic light": "Світлофор",
        "traffic sign": "Дорожній знак",
        "vegetation": "Рослинність",
        "terrain": "Ґрунт",
        "sky": "Небо",
        "person": "Людина",
        "rider": "Вершник",
        "car": "Автомобіль",
        "truck": "Вантажівка",
        "bus": "Автобус",
        "train": "Поїзд",
        "motorcycle": "Мотоцикл",
        "bicycle": "Велосипед",
        "Master's Diploma Project": "Кваліфікаційний магістерський проєкт"
    };


    Object.assign(uk, {
        "Image Segmentation": "Сегментація зображень",
        "Upload an image": "Завантажити зображення",
        "Upload an image captured by the robot camera and analyze its semantic structure using the semantic segmentation model.": "Завантажте зображення, отримане з камери робота, та проаналізуйте його семантичну структуру за допомогою моделі семантичної сегментації.",
        "Click to browse or drag and drop an image here": "Натисніть, щоб обрати файл, або перетягніть зображення сюди",
        "JPG, JPEG or PNG": "JPG, JPEG або PNG",
        "Selected file:": "Обраний файл:",
        "Run segmentation": "Запустити сегментацію",
        "Segmentation Result": "Результат сегментації",
        "Comparison between the original image and the generated semantic segmentation result.": "Порівняння оригінального зображення з результатом семантичної сегментації.",
        "Original image": "Оригінальне зображення",
        "Segmentation overlay": "Накладання сегментації",
        "Input": "Вхідне зображення",
        "Prediction": "Результат прогнозування",
        "Download Overlay": "Завантажити накладання",
        "Download Mask": "Завантажити маску",
        "Original resolution": "Оригінальна роздільна здатність",
        "Input tensor": "Вхідний тензор",
        "Segmentation model": "Модель сегментації",
        "Compute device": "Обчислювальний пристрій",
        "Segmentation Analytics": "Аналітика сегментації",
        "Inference time": "Час обробки",
        "Image size": "Розмір зображення",
        "Pixels analyzed": "Проаналізовано пікселів",
        "Classes detected": "Виявлено класів",
        "Input file size": "Розмір вхідного файлу",
        "Inference throughput": "Швидкість обробки",
        "Dominant class": "Домінантний клас",
        "Top 3 classes coverage": "Покриття трьох основних класів",
        "Class Distribution": "Розподіл класів",
        "Scene Summary": "Підсумок аналізу сцени",
        "The dominant class is": "Домінантний клас —",
        "covering": "охоплює",
        "of the image.": "зображення.",
        "The three largest classes are": "Три найбільші класи:",
        "A total of": "Загалом виявлено",
        "classes were detected.": "класів.",
        "Inference time:": "Час обробки:",
        "Master's Diploma Project": "Кваліфікаційний магістерський проєкт",
        "Semantic Segmentation Robot": "Система семантичної сегментації",
        "Mask": "Маска",
        "building": "будівля",
        "vegetation": "рослинність",
        "road": "дорога",
        "sidewalk": "тротуар",
        "sky": "небо",
        "truck": "вантажівка",
        "terrain": "ґрунт",
        "car": "автомобіль",
        "wall": "стіна",
        "motorcycle": "мотоцикл",
        "traffic sign": "дорожній знак",
        "person": "людина"
    });

    const en = Object.fromEntries(
        Object.entries(uk).map(([english, ukrainian]) => [ukrainian, english])
    );

    let language = "en";

    // Зберігаємо початковий текст, щоб не перекладати переклад повторно.
    const originalTexts = new WeakMap();
    const originalAttributes = new WeakMap();

    function translate(value) {
        const dictionary = language === "uk" ? uk : en;
        const keys = Object.keys(dictionary).sort((a, b) => b.length - a.length);
        let result = value;

        for (const key of keys) {
            result = result.split(key).join(dictionary[key]);
        }

        return result;
    }

    function updatePage(root = document.body) {
        const description = document.querySelector(".page-title p");
        if (description) {
            const english = "Upload an image captured by the robot camera and analyze its semantic structure using the semantic segmentation model.";
            const ukrainian = "Завантажте зображення, отримане з камери робота, та проаналізуйте його семантичну структуру за допомогою моделі семантичної сегментації.";
            const desiredText = language === "uk" ? ukrainian : english;

            if (description.textContent.trim() !== desiredText) {
                description.textContent = desiredText;
            }
        }
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
        const nodes = [];

        while (walker.nextNode()) nodes.push(walker.currentNode);

        nodes.forEach(node => {
            if (node.parentElement?.closest(".language-switch, .page-title p")) return;

            if (!originalTexts.has(node)) {
                originalTexts.set(node, node.nodeValue);
            }

            const original = originalTexts.get(node);
            const normalized = original.replace(/\s+/g, " ").trim();

            if (normalized === "Upload an image captured by the robot camera and") {
                node.nodeValue = language === "uk"
                    ? "Завантажте зображення, отримане з камери робота, та"
                    : original;
            } else if (normalized === "analyze its semantic structure using the semantic") {
                node.nodeValue = language === "uk"
                    ? "проаналізуйте його семантичну структуру за допомогою моделі"
                    : original;
            } else if (normalized === "segmentation model.") {
                node.nodeValue = language === "uk"
                    ? "семантичної сегментації."
                    : original;
            } else {
                node.nodeValue = translate(originalTexts.get(node));
            }
        });

        root.querySelectorAll("[placeholder], [title], [alt]").forEach(element => {
            let saved = originalAttributes.get(element);

            if (!saved) {
                saved = {};
                originalAttributes.set(element, saved);
            }

            ["placeholder", "title", "alt"].forEach(attribute => {
                if (!element.hasAttribute(attribute)) return;

                if (!(attribute in saved)) {
                    saved[attribute] = element.getAttribute(attribute);
                }

                element.setAttribute(attribute, translate(saved[attribute]));
            });
        });

        document.documentElement.lang = language;
        document.getElementById("lang-uk")?.classList.toggle("active", language === "uk");
        document.getElementById("lang-en")?.classList.toggle("active", language === "en");
    }

    document.getElementById("lang-uk")?.addEventListener("click", () => {
        language = "uk";
        updatePage();
    });

    document.getElementById("lang-en")?.addEventListener("click", () => {
        language = "en";
        updatePage();
    });

    updatePage();

    new MutationObserver(() => updatePage()).observe(document.body, {
        childList: true,
        subtree: true
    });
})();
