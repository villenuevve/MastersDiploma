(function () {
    const translations = {
        "Semantic Segmentation Robot": "Система семантичної сегментації",
        "Semantic Segmentation": "Семантична сегментація",
        "Upload Image": "Завантажити зображення",
        "Choose File": "Обрати файл",
        "Drag and drop an image here": "Перетягніть зображення сюди",
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
        "Loading...": "Завантаження...",
        "Error": "Помилка",
        "Success": "Успішно",
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
        "Upload an image": "Завантажте зображення",
        "Segmentation completed successfully": "Сегментацію успішно завершено",
        "Invalid image. Please upload a valid JPG or PNG.": "Некоректне зображення. Завантажте файл JPG або PNG.",
        "The uploaded file is empty.": "Завантажений файл порожній.",
        "Image is too large. Maximum size is 20 MB.": "Зображення завелике. Максимальний розмір — 20 МБ."
    };

    function translateText(text) {
        const trimmed = text.trim();
        return translations[trimmed] || text;
    }

    function translatePage(root = document.body) {
        const walker = document.createTreeWalker(
            root,
            NodeFilter.SHOW_TEXT
        );

        const nodes = [];
        while (walker.nextNode()) {
            nodes.push(walker.currentNode);
        }

        nodes.forEach(node => {
            const original = node.nodeValue;
            if (original && original.trim()) {
                const translated = translateText(original);
                if (translated !== original.trim()) {
                    node.nodeValue = original.replace(original.trim(), translated);
                }
            }
        });

        root.querySelectorAll("input[placeholder], textarea[placeholder], button[title], img[alt]").forEach(element => {
            ["placeholder", "title", "alt"].forEach(attribute => {
                if (element.hasAttribute(attribute)) {
                    const value = element.getAttribute(attribute);
                    element.setAttribute(attribute, translateText(value));
                }
            });
        });
    }

    window.translateSegmentationText = translateText;
    window.translateSegmentationPage = translatePage;

    document.addEventListener("DOMContentLoaded", () => {
        translatePage();

        const observer = new MutationObserver(() => {
            translatePage();
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    });
})();
