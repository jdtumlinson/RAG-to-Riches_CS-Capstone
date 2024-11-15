function uploadDocument(input) {
    const file = input.files[0];
    if (!file) return;

    const documentBox = input.closest('.document-box');
    const imgElement = documentBox.querySelector('img');

    // Clear any existing previews
    documentBox.querySelectorAll('.doc-preview, canvas').forEach(el => el.remove());

    // Handle PDF files
    if (file.type === 'application/pdf') {
        const fileReader = new FileReader();
        fileReader.onload = function () {
            const typedarray = new Uint8Array(this.result);

            // Load PDF and render the first page
            pdfjsLib.getDocument(typedarray).promise.then(pdf => {
                pdf.getPage(1).then(page => {
                    const scale = 1.5;
                    const viewport = page.getViewport({ scale: scale });
                    const canvas = document.createElement('canvas');
                    const context = canvas.getContext('2d');
                    canvas.height = viewport.height;
                    canvas.width = viewport.width;

                    page.render({ canvasContext: context, viewport: viewport }).promise.then(() => {
                        imgElement.style.display = 'none'; // Hide img
                        documentBox.appendChild(canvas); // Append canvas in place of img
                    });
                });
            });
        };
        fileReader.readAsArrayBuffer(file);

    // Handle DOCX files
    } else if (file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
        const reader = new FileReader();
        reader.onload = function (e) {
            const arrayBuffer = e.target.result;
            mammoth.convertToHtml({ arrayBuffer: arrayBuffer })
                .then(result => {
                    imgElement.style.display = 'none'; // Hide img
                    const docPreview = document.createElement("div");
                    docPreview.classList.add("doc-preview");
                    docPreview.innerHTML = result.value; // Display DOCX content as HTML
                    documentBox.appendChild(docPreview); // Append preview in place of img
                })
                .catch(err => console.error(err));
        };
        reader.readAsArrayBuffer(file);

    // Handle Image Files
    } else if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = function (e) {
            imgElement.style.display = 'block';
            imgElement.src = e.target.result; // Set the img src directly
        };
        reader.readAsDataURL(file);

    } else {
        alert("Unsupported file type. Please upload a PDF, DOCX, or image file.");
    }
}