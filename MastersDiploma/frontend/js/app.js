/* ========= 
       API
    ======== */

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

    const overlayButton =
        document.getElementById("overlayButton");

    const maskButton =
        document.getElementById("maskButton");

    const downloadOverlay =
        document.getElementById("downloadOverlay");

    const downloadMask =
        document.getElementById("downloadMask");

    const classesContainer =
        document.getElementById("classes");

    const classCount =
        document.getElementById("classCount");


    let selectedImage = null;

    let overlayUrl = "";

    let maskUrl = "";


    /* =============
       FILE INPUT
    ================ */

    fileInput.addEventListener(
        "change",
        function () {

            if (
                !this.files ||
                this.files.length === 0
            ) {
                return;
            }

            selectImage(
                this.files[0]
            );

        }
    );


    function selectImage(file) {

        if (
            !file.type.startsWith("image/")
        ) {

            showError(
                "Please select a valid image file."
            );

            return;
        }


        selectedImage =
            file;


        selectedFile.style.display =
            "block";


        selectedFile.innerHTML =
            `<strong>Selected file:</strong> ${file.name}`;


        predictButton.disabled =
            false;


        errorBox.style.display =
            "none";


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

        overlayButton.classList.remove("active");
        maskButton.classList.remove("active");

    }


    /* =============
       DRAG & DROP
    =============== */

    uploadArea.addEventListener(
        "dragover",
        function (event) {

            event.preventDefault();

            uploadArea.classList.add(
                "dragover"
            );

        }
    );


    uploadArea.addEventListener(
        "dragleave",
        function () {

            uploadArea.classList.remove(
                "dragover"
            );

        }
    );


    uploadArea.addEventListener(
        "drop",
        function (event) {

            event.preventDefault();

            uploadArea.classList.remove(
                "dragover"
            );


            const files =
                event.dataTransfer.files;


            if (
                !files ||
                files.length === 0
            ) {
                return;
            }


            selectImage(
                files[0]
            );

        }
    );


    /* ========
       PREDICT
    ========= */

    predictButton.addEventListener(
        "click",
        async function () {

            if (!selectedImage) {

                showError(
                    "Please select an image first."
                );

                return;
            }


            predictButton.disabled =
                true;

            loading.style.display =
                "flex";

            errorBox.style.display =
                "none";

            try {

                const data =
                    await window.SegmentationModel.predict(
                        selectedImage,
                        API_URL
                    );


                console.log(
                    "Prediction:",
                    data
                );


                /* =========================
                   RESULT FILES
                ========================= */

                const overlayFilename =
                    data.overlay_file
                        .split("/")
                        .pop();


                const maskFilename =
                    data.mask_file
                        .split("/")
                        .pop();


                const fullOverlayUrl =
                    `${API_URL}/results/${overlayFilename}`;

                overlayUrl = data.preview_url
                    ? `${API_URL}${data.preview_url}`
                    : fullOverlayUrl;


                maskUrl =
                    `${API_URL}/results/${maskFilename}`;



                window.SegmentationView.showImage(
                    resultImage,
                    `${overlayUrl}?t=${Date.now()}`
                );


                resultTitle.textContent =
                    "Segmentation overlay";


                overlayButton.classList.add(
                    "active"
                );


                maskButton.classList.remove(
                    "active"
                );

                downloadOverlay.href =
                    fullOverlayUrl;

                downloadOverlay.download =
                    overlayFilename;


                downloadMask.href =
                    maskUrl;

                downloadMask.download =
                    maskFilename;

                window.SegmentationView.renderImageInformation(data);


                window.SegmentationView.renderDetectedClasses(
                    data.detected_classes,
                    classesContainer,
                    classCount,
                    CITYSCAPES_COLORS
                );

                window.dispatchEvent(
                    new CustomEvent("segmentation-result", {
                        detail: data
                    })
                );


                window.SegmentationView.showResultSection(resultSection);


                resultSection.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });

            }

            catch (error) {
                console.error("Segmentation failed:", error);
                showError(error.message || String(error));
            }

            finally {

                predictButton.disabled =
                    false;

                loading.style.display =
                    "none";

            }

        }
    );


    /* =====================
       IMAGE MODE CONTROLS
    ======================== */

    overlayButton.addEventListener("click", function () {
        if (!overlayUrl) {
            return;
        }

        window.SegmentationView.showImageMode(
            resultImage,
            resultTitle,
            overlayButton,
            maskButton,
            overlayUrl,
            "Segmentation overlay"
        );
    });

    maskButton.addEventListener("click", function () {
        if (!maskUrl) {
            return;
        }

        window.SegmentationView.showImageMode(
            resultImage,
            resultTitle,
            maskButton,
            overlayButton,
            maskUrl,
            "Segmentation mask"
        );
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
