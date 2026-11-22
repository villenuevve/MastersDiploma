(() => {
    "use strict";

    const resultSection = document.getElementById("resultSection");
    const resultLinks = document.querySelectorAll(
        ".primary-nav [data-requires-results]"
    );

    if (!resultSection || resultLinks.length === 0) return;

    function syncResultNavigation() {
        const hasResults = !resultSection.hidden &&
            window.getComputedStyle(resultSection).display !== "none";

        resultLinks.forEach(link => {
            link.hidden = !hasResults;
        });
    }

    const observer = new MutationObserver(syncResultNavigation);

    observer.observe(resultSection, {
        attributes: true,
        attributeFilter: ["style", "hidden", "class"]
    });

    syncResultNavigation();
})();
