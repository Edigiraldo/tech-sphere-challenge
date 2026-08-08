/**
 * admin.js — Tech Sphere Challenge administration console.
 *
 * Manages the clinical document lifecycle from the browser: PDF upload with
 * validation, document listing with status filter and auto-refresh polling,
 * and document deletion with confirmation.
 *
 * Communicates exclusively with the backend REST API:
 *   POST   /documents           — upload a PDF
 *   GET    /documents?status=X   — list documents
 *   DELETE /documents/{id}       — delete a document
 *
 * No framework dependencies — vanilla JS matching the existing frontend pattern.
 */

document.addEventListener("DOMContentLoaded", () => {

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    /** Polling interval in milliseconds. */
    const POLL_INTERVAL_MS = 5000;

    /** Maximum upload size in bytes (20 MB). */
    const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;

    // -----------------------------------------------------------------------
    // DOM references — Upload
    // -----------------------------------------------------------------------
    const uploadFileInput = document.getElementById("upload-file-input");
    const uploadBtn = document.getElementById("upload-btn");
    const uploadStatus = document.getElementById("upload-status");
    const uploadError = document.getElementById("upload-error");

    // -----------------------------------------------------------------------
    // DOM references — Document list
    // -----------------------------------------------------------------------
    const statusFilter = document.getElementById("status-filter");
    const refreshBtn = document.getElementById("refresh-btn");
    const listError = document.getElementById("list-error");
    const documentTbody = document.getElementById("document-tbody");
    const documentsCount = document.getElementById("documents-count");

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    let pollTimerId = null;

    // -----------------------------------------------------------------------
    // Utility helpers
    // -----------------------------------------------------------------------

    /**
     * Escape HTML special characters to prevent DOM injection.
     * @param {string} str
     * @returns {string}
     */
    function escapeHtml(str) {
        const div = document.createElement("div");
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }

    /**
     * Format file size in human-readable form (KB or MB).
     * @param {number} bytes
     * @returns {string}
     */
    function formatSize(bytes) {
        if (bytes < 1024) return `${bytes} B`;
        const kb = bytes / 1024;
        if (kb < 1024) return `${kb.toFixed(1)} KB`;
        const mb = kb / 1024;
        return `${mb.toFixed(1)} MB`;
    }

    /**
     * Format an ISO-8601 UTC timestamp to a short locale string.
     * @param {string} isoStr
     * @returns {string}
     */
    function formatDate(isoStr) {
        try {
            const d = new Date(isoStr);
            if (isNaN(d.getTime())) return isoStr;
            return d.toLocaleString("es-CO", {
                year: "numeric",
                month: "2-digit",
                day: "2-digit",
                hour: "2-digit",
                minute: "2-digit",
            });
        } catch (_e) {
            return isoStr;
        }
    }

    /** Hide an element. */
    function hide(el) { el.classList.add("hidden"); }

    /** Show an element. */
    function show(el) { el.classList.remove("hidden"); }

    // -----------------------------------------------------------------------
    // Status badge rendering
    // -----------------------------------------------------------------------

    /**
     * Return an HTML string for a coloured status badge.
     * @param {string} status
     * @returns {string}
     */
    function renderStatusBadge(status) {
        const labelMap = {
            pending: "Pendiente",
            processing: "Procesando",
            ready: "Listo",
            failed: "Fallido",
            deleted: "Eliminado",
        };
        // Use a hardcoded safe mapping for CSS classes to prevent DOM injection.
        // Only known status values are interpolated into the class attribute;
        // unrecognised values render with a safe fallback class.
        const safeStatus = labelMap.hasOwnProperty(status) ? status : "unknown";
        const cssClass = `status-badge status-badge-${safeStatus}`;
        const label = labelMap[status] || status;
        return `<span class="${cssClass}">${escapeHtml(label)}</span>`;
    }

    // -----------------------------------------------------------------------
    // Upload logic
    // -----------------------------------------------------------------------

    /** Enable or disable the upload button based on file selection. */
    function updateUploadButton() {
        uploadBtn.disabled = !uploadFileInput.files || uploadFileInput.files.length === 0;
    }

    /**
     * Clear upload error and status messages.
     */
    function clearUploadMessages() {
        hide(uploadError);
        hide(uploadStatus);
        uploadError.textContent = "";
        uploadStatus.textContent = "";
    }

    /**
     * Show a temporary inline status message.
     * @param {string} text
     * @param {boolean} isError
     */
    function showUploadStatus(text, isError) {
        uploadStatus.textContent = text;
        uploadStatus.className = isError
            ? "inline-status inline-status-error"
            : "inline-status inline-status-success";
        show(uploadStatus);

        // Auto-hide after 6 seconds (except errors, which persist longer)
        if (!isError) {
            setTimeout(() => { hide(uploadStatus); }, 6000);
        }
    }

    /**
     * Show a persistent error in the upload card.
     * @param {string} text
     */
    function showUploadError(text) {
        uploadError.textContent = text;
        show(uploadError);
    }

    /**
     * Validate the selected file before upload.
     * @returns {string|null} Error message or null if valid.
     */
    function validateUpload() {
        const file = uploadFileInput.files[0];
        if (!file) return "Seleccione un archivo.";

        if (!file.name.toLowerCase().endsWith(".pdf")) {
            return "Solo se aceptan archivos PDF.";
        }

        if (file.size === 0) {
            return "El archivo está vacío.";
        }

        if (file.size > MAX_UPLOAD_BYTES) {
            return `El archivo excede el tamaño máximo de ${formatSize(MAX_UPLOAD_BYTES)}.`;
        }

        return null;
    }

    /**
     * Upload the selected PDF file via POST /documents.
     */
    async function handleUpload() {
        clearUploadMessages();

        const validationError = validateUpload();
        if (validationError) {
            showUploadError(validationError);
            return;
        }

        const file = uploadFileInput.files[0];
        const formData = new FormData();
        formData.append("file", file);

        uploadBtn.disabled = true;
        showUploadStatus("Subiendo y procesando documento…", false);

        try {
            const response = await fetch("/documents", {
                method: "POST",
                body: formData,
            });

            const body = await response.json();

            if (!response.ok) {
                const detail = body.detail || `Error del servidor (${response.status})`;
                showUploadError(detail);
                showUploadStatus("Error al subir el documento.", true);
                return;
            }

            // Success — show result and refresh list
            const msg = body.status === "ready"
                ? `Documento "${escapeHtml(body.filename)}" procesado exitosamente.`
                : `Documento "${escapeHtml(body.filename)}" cargado (estado: ${escapeHtml(body.status)}).`;
            showUploadStatus(msg, body.status !== "ready");

            // Reset file input so the same file can be re-uploaded
            uploadFileInput.value = "";
            updateUploadButton();

            // Refresh the document list immediately
            await fetchDocuments();

        } catch (err) {
            showUploadError(`Error de red: ${err.message}`);
            showUploadStatus("Error de conexión al subir el documento.", true);
        } finally {
            uploadBtn.disabled = !uploadFileInput.files || uploadFileInput.files.length === 0;
        }
    }

    // -----------------------------------------------------------------------
    // Document list logic
    // -----------------------------------------------------------------------

    /**
     * Fetch the document list from GET /documents and render the table.
     */
    async function fetchDocuments() {
        hide(listError);

        const selectedStatus = statusFilter.value;
        let url = "/documents";
        if (selectedStatus) {
            url += `?status=${encodeURIComponent(selectedStatus)}`;
        }

        try {
            const response = await fetch(url);

            if (!response.ok) {
                const body = await response.json().catch(() => ({}));
                const detail = body.detail || `Error del servidor (${response.status})`;
                listError.textContent = detail;
                show(listError);
                renderEmptyTable("Error al cargar documentos.");
                return;
            }

            const data = await response.json();
            const docs = data.documents || [];

            if (docs.length === 0) {
                const filterLabel = selectedStatus
                    ? `No hay documentos con estado "${selectedStatus}".`
                    : "No hay documentos subidos.";
                renderEmptyTable(filterLabel);
                documentsCount.textContent = "0 documentos";
                show(documentsCount);
                return;
            }

            renderDocuments(docs);
            documentsCount.textContent = `${docs.length} documento${docs.length !== 1 ? "s" : ""}`;
            show(documentsCount);

        } catch (err) {
            listError.textContent = `Error de red: ${err.message}`;
            show(listError);
            renderEmptyTable("Error de conexión al cargar documentos.");
        }
    }

    /**
     * Render the document table rows.
     * @param {Array} docs — Array of document response objects.
     */
    function renderDocuments(docs) {
        documentTbody.innerHTML = "";

        docs.forEach((doc) => {
            const tr = document.createElement("tr");

            // Filename
            const tdName = document.createElement("td");
            tdName.className = "doc-name";
            tdName.textContent = doc.filename;
            tr.appendChild(tdName);

            // Status
            const tdStatus = document.createElement("td");
            tdStatus.innerHTML = renderStatusBadge(doc.status);
            tr.appendChild(tdStatus);

            // Size
            const tdSize = document.createElement("td");
            tdSize.className = "doc-size";
            tdSize.textContent = formatSize(doc.size_bytes);
            tr.appendChild(tdSize);

            // Uploaded date
            const tdDate = document.createElement("td");
            tdDate.className = "doc-date";
            tdDate.textContent = formatDate(doc.uploaded_at);
            tr.appendChild(tdDate);

            // Actions
            const tdActions = document.createElement("td");
            tdActions.className = "doc-actions";

            // Show delete button for non-deleted documents
            if (doc.status !== "deleted") {
                const deleteBtn = document.createElement("button");
                deleteBtn.className = "btn btn-danger btn-sm";
                deleteBtn.textContent = "Eliminar";
                deleteBtn.title = `Eliminar "${doc.filename}" y sus fragmentos indexados`;
                deleteBtn.addEventListener("click", () => {
                    handleDelete(doc);
                });
                tdActions.appendChild(deleteBtn);
            } else {
                tdActions.textContent = "—";
            }

            tr.appendChild(tdActions);
            documentTbody.appendChild(tr);
        });
    }

    /**
     * Render the empty-table placeholder row.
     * @param {string} message
     */
    function renderEmptyTable(message) {
        documentTbody.innerHTML = "";
        const tr = document.createElement("tr");
        tr.className = "table-empty-row";
        const td = document.createElement("td");
        td.colSpan = 5;
        td.className = "placeholder-text";
        td.textContent = message;
        tr.appendChild(td);
        documentTbody.appendChild(tr);
        hide(documentsCount);
    }

    // -----------------------------------------------------------------------
    // Delete logic
    // -----------------------------------------------------------------------

    /**
     * Handle deletion of a document with confirmation.
     * @param {object} doc — Document response object with at least .document_id and .filename.
     */
    async function handleDelete(doc) {
        const confirmed = confirm(
            `¿Eliminar "${doc.filename}"?\n\n` +
            "Esta acción eliminará el documento y todos sus fragmentos indexados " +
            "de la base de conocimiento. No se puede deshacer."
        );
        if (!confirmed) return;

        hide(listError);

        try {
            const response = await fetch(`/documents/${encodeURIComponent(doc.document_id)}`, {
                method: "DELETE",
            });

            if (!response.ok) {
                const body = await response.json().catch(() => ({}));
                const detail = body.detail || `Error del servidor (${response.status})`;
                listError.textContent = `Error al eliminar: ${detail}`;
                show(listError);
                return;
            }

            // Refresh the list
            await fetchDocuments();

        } catch (err) {
            listError.textContent = `Error de red al eliminar: ${err.message}`;
            show(listError);
        }
    }

    // -----------------------------------------------------------------------
    // Polling
    // -----------------------------------------------------------------------

    /** Start periodic polling. */
    function startPolling() {
        stopPolling();
        pollTimerId = setInterval(fetchDocuments, POLL_INTERVAL_MS);
    }

    /** Stop periodic polling. */
    function stopPolling() {
        if (pollTimerId !== null) {
            clearInterval(pollTimerId);
            pollTimerId = null;
        }
    }

    // -----------------------------------------------------------------------
    // Event listeners
    // -----------------------------------------------------------------------
    uploadFileInput.addEventListener("change", () => {
        clearUploadMessages();
        updateUploadButton();
    });

    uploadBtn.addEventListener("click", handleUpload);

    statusFilter.addEventListener("change", () => {
        fetchDocuments();
    });

    refreshBtn.addEventListener("click", () => {
        fetchDocuments();
    });

    // -----------------------------------------------------------------------
    // Initialisation
    // -----------------------------------------------------------------------
    updateUploadButton();
    fetchDocuments();
    startPolling();

    // Pause polling when the page is hidden, resume when visible
    document.addEventListener("visibilitychange", () => {
        if (document.hidden) {
            stopPolling();
        } else {
            fetchDocuments();
            startPolling();
        }
    });
});
