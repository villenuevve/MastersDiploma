(function () {
    "use strict";

    async function predict(file, apiUrl) {
        const formData = new FormData();
        formData.append("file", file);

        let response;

        try {
            response = await fetch(`${apiUrl}/predict`, {
                method: "POST",
                body: formData
            });
        } catch (error) {
            throw new Error(
                `Cannot connect to API ${apiUrl}/predict: ${error.message}`
            );
        }

        const responseText = await response.text();

        if (!response.ok) {
            throw new Error(
                `API error ${response.status}: ${responseText}`
            );
        }

        try {
            return JSON.parse(responseText);
        } catch {
            throw new Error(
                `API returned invalid JSON: ${responseText.slice(0, 500)}`
            );
        }
    }

    window.SegmentationModel = { predict };
})();
