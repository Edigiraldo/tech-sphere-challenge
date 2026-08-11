/**
 * summary.js — Tech Sphere Challenge call summary page.
 *
 * Reads `call_id` from the URL query parameter, fetches the call summary
 * from GET /calls/{call_id}/summary, and renders the structured summary
 * with XSS-safe text, traceable source citations, and severity indicators.
 */

document.addEventListener("DOMContentLoaded", () => {

    // -----------------------------------------------------------------------
    // DOM references
    // -----------------------------------------------------------------------
    const summaryContainer = document.getElementById("summary-container");
    const summaryError = document.getElementById("summary-error");
    const sourcesSection = document.getElementById("sources-section");
    const sourcesList = document.getElementById("sources-list");
    const summaryBadge = document.getElementById("summary-badge");

    // -----------------------------------------------------------------------
    // Read call_id from URL
    // -----------------------------------------------------------------------
    const urlParams = new URLSearchParams(window.location.search);
    const callId = urlParams.get("call_id");

    if (!callId) {
        showError("No se especificó un identificador de llamada. "
            + "Use ?call_id=... en la URL.");
        return;
    }

    // -----------------------------------------------------------------------
    // Fetch and render
    // -----------------------------------------------------------------------
    fetchSummary(callId);

    /**
     * Fetch the summary from the API and render it.
     * @param {string} callId
     */
    async function fetchSummary(callId) {
        setLoading(true);

        try {
            const response = await fetch(`/calls/${encodeURIComponent(callId)}/summary`);

            if (!response.ok) {
                let detail = "No se pudo obtener el resumen de la llamada.";
                try {
                    const err = await response.json();
                    detail = err.detail || detail;
                } catch (_) { /* use default */ }
                showError(detail);
                return;
            }

            const data = await response.json();
            renderSummary(data);
        } catch (err) {
            showError("Error de conexión al obtener el resumen: " + err.message);
        }
    }

    /**
     * Render the structured summary into the page.
     * @param {object} data - SummaryResponse from the API
     */
    function renderSummary(data) {
        // Update badge
        if (summaryBadge) {
            summaryBadge.textContent = "Completada";
            summaryBadge.classList.remove("state-idle");
            summaryBadge.classList.add("state-ended");
        }

        // Sections to render in order
        const sections = [
            { heading: "Paciente", content: data.patient_summary, icon: "&#x1F464;" },
            { heading: "Procedimiento", content: data.procedure_summary, icon: "&#x1F3E5;" },
            { heading: "Síntomas", content: formatSymptoms(data.symptoms_summary), icon: "&#x1F3A4;" },
            {
                heading: "Decisión de Escalamiento",
                content: data.decision_summary,
                icon: "&#x26A0;",
                severityClass: getSeverityClass(data.decision_summary),
            },
            { heading: "Próximos Pasos", content: data.next_steps, icon: "&#x27A1;" },
        ];

        let html = "";

        // Metadata row
        html += '<div class="summary-meta">';
        html += `<span class="summary-meta-item"><strong>Llamada:</strong> ${escapeHtml(data.call_id)}</span>`;
        if (data.created_at) {
            const created = new Date(data.created_at);
            html += `<span class="summary-meta-item"><strong>Fecha:</strong> ${created.toLocaleString("es-CO")}</span>`;
        }
        html += '</div>';

        // Sections
        html += '<div class="summary-sections">';
        sections.forEach((sec) => {
            const sevClass = sec.severityClass || "";
            html += '<div class="summary-section' + (sevClass ? ' ' + sevClass : '') + '">';
            html += `<h3 class="summary-section-heading">${sec.icon} ${escapeHtml(sec.heading)}</h3>`;
            html += `<p class="summary-section-content">${escapeHtml(sec.content)}</p>`;
            html += '</div>';
        });
        html += '</div>';

        summaryContainer.innerHTML = html;

        // Render sources
        if (data.sources && data.sources.length > 0) {
            renderSources(data.sources);
        } else {
            sourcesSection.classList.add("hidden");
        }
    }

    /**
     * Render the list of traceable source citations.
     * @param {Array<{document_id: string, source_filename: string, page_number: number}>} sources
     */
    function renderSources(sources) {
        sourcesSection.classList.remove("hidden");

        let html = '<ul class="sources-list-items">';
        sources.forEach((src) => {
            html += '<li class="source-item">';
            html += '<span class="source-icon">&#x1F4C4;</span>';
            html += '<span class="source-filename">' + escapeHtml(src.source_filename || src.document_id) + '</span>';
            if (src.page_number && src.page_number > 0) {
                html += '<span class="source-page">p. ' + escapeHtml(String(src.page_number)) + '</span>';
            }
            html += '<span class="source-id">ID: ' + escapeHtml(src.document_id) + '</span>';
            html += '</li>';
        });
        html += '</ul>';

        sourcesList.innerHTML = html;
    }

    /**
     * Format the symptoms summary — may be multi-line (one line per domain).
     * @param {string} text
     * @returns {string} HTML-safe text with visible line separation
     */
    function formatSymptoms(text) {
        if (!text) return "";
        return text;
    }

    /**
     * Derive a CSS severity class from the decision text.
     * @param {string} text
     * @returns {string} CSS class name or empty string
     */
    function getSeverityClass(text) {
        if (!text) return "";
        const lower = text.toLowerCase();
        if (lower.includes("rojo") || lower.includes("inmediato")) {
            return "summary-severity-red";
        }
        if (lower.includes("amarillo") || lower.includes("precaucion") || lower.includes("precaución")) {
            return "summary-severity-yellow";
        }
        if (lower.includes("verde") || lower.includes("normal")) {
            return "summary-severity-green";
        }
        return "";
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    function setLoading(visible) {
        if (visible) {
            summaryContainer.innerHTML =
                '<p class="placeholder-text summary-loading">Cargando resumen...</p>';
        }
    }

    function showError(message) {
        summaryContainer.innerHTML =
            '<p class="placeholder-text">No se pudo cargar el resumen.</p>';
        summaryError.textContent = message;
        summaryError.classList.remove("hidden");
    }

    function escapeHtml(str) {
        if (!str) return "";
        const div = document.createElement("div");
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }
});
