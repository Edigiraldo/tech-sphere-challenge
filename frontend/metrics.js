/**
 * Metrics Dashboard — Vanilla JS client for the /metrics endpoints.
 *
 * Provides a read-only view of:
 *   - Global summary (GET /metrics/summary)
 *   - Calls list   (GET /metrics/calls)
 *   - Call detail  (GET /metrics/calls/{call_id})
 *
 * All data is fetched client-side from the API.  No frameworks.
 */
(function () {
    "use strict";

    // -----------------------------------------------------------------------
    // DOM references
    // -----------------------------------------------------------------------

    const summaryContent = document.getElementById("summary-content");
    const callsContent = document.getElementById("calls-content");
    const callsSection = document.getElementById("calls-section");
    const detailSection = document.getElementById("detail-section");
    const detailTitle = document.getElementById("detail-title");
    const detailContent = document.getElementById("detail-content");
    const detailBackLink = document.getElementById("detail-back-link");

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /** Format a nullable number for display. */
    function fmt(value, suffix) {
        if (value === null || value === undefined) {
            return '<span class="metric-null">—</span>';
        }
        var s = suffix || "";
        return Number(value).toLocaleString() + s;
    }

    /** Format a nullable float with fixed decimals. */
    function fmtFloat(value, decimals) {
        if (value === null || value === undefined) {
            return '<span class="metric-null">—</span>';
        }
        var d = decimals || 2;
        return Number(value).toFixed(d);
    }

    /** Format cost (USD). */
    function fmtCost(value) {
        if (value === null || value === undefined) {
            return '<span class="metric-null">—</span>';
        }
        return "$" + Number(value).toFixed(6);
    }

    // -----------------------------------------------------------------------
    // Fetch helpers
    // -----------------------------------------------------------------------

    async function fetchJSON(url) {
        var resp = await fetch(url);
        if (!resp.ok) {
            throw new Error("HTTP " + resp.status + " from " + url);
        }
        return resp.json();
    }

    // -----------------------------------------------------------------------
    // Render: Summary
    // -----------------------------------------------------------------------

    function renderSummary(data) {
        var html = "";

        // Top-level counts
        html += '<div class="metrics-grid">';
        html += metricBox("Llamadas", fmt(data.call_count));
        html += metricBox("Turnos Totales", fmt(data.total_turns));
        html += metricBox("Tokens Entrada", fmt(data.total_input_tokens));
        html += metricBox("Tokens Salida", fmt(data.total_output_tokens));
        html += metricBox("Consultas RAG", fmt(data.total_rag_queries));
        html += metricBox("Llamadas al Modelo", fmt(data.total_model_calls));
        html += metricBox("Costo Estimado", fmtCost(data.total_estimated_cost_usd));
        html += "</div>";

        // Percentiles — latency
        html += "<h3 style='margin-bottom:0.5rem;color:#334155'>Latencias (ms)</h3>";
        html += '<div class="percentile-grid">';
        html += pctBox("P50", fmtFloat(data.latency_p50_ms, 1));
        html += pctBox("P95", fmtFloat(data.latency_p95_ms, 1));
        html += "</div>";

        // Percentiles — components
        html += "<h3 style='margin-bottom:0.5rem;color:#334155'>Componentes (ms)</h3>";
        html += '<div class="percentile-grid">';
        html += pctBox("TTS P50", fmtFloat(data.tts_p50_ms, 1));
        html += pctBox("TTS P95", fmtFloat(data.tts_p95_ms, 1));
        html += pctBox("STT P50", fmtFloat(data.stt_p50_ms, 1));
        html += pctBox("STT P95", fmtFloat(data.stt_p95_ms, 1));
        html += pctBox("LLM P50", fmtFloat(data.llm_p50_ms, 1));
        html += pctBox("LLM P95", fmtFloat(data.llm_p95_ms, 1));
        html += "</div>";

        summaryContent.innerHTML = html;
    }

    function metricBox(label, value) {
        return (
            '<div class="metric-box">' +
            '<div class="metric-value">' +
            value +
            "</div>" +
            '<div class="metric-label">' +
            label +
            "</div>" +
            "</div>"
        );
    }

    function pctBox(label, value) {
        return (
            '<div class="percentile-box">' +
            '<div class="pct-label">' +
            label +
            "</div>" +
            '<div class="pct-value">' +
            value +
            " ms</div>" +
            "</div>"
        );
    }

    // -----------------------------------------------------------------------
    // Render: Calls list
    // -----------------------------------------------------------------------

    function renderCallsList(data) {
        var calls = data.calls || [];
        if (calls.length === 0) {
            callsContent.innerHTML =
                '<p class="placeholder-text">No hay llamadas completadas registradas.</p>';
            return;
        }

        var html = '<table class="calls-table">';
        html += "<thead><tr>";
        html += "<th>Call ID</th>";
        html += "<th>Paciente</th>";
        html += "<th>Turnos</th>";
        html += "<th>Latencia (ms)</th>";
        html += "<th>Tokens In</th>";
        html += "<th>Tokens Out</th>";
        html += "<th>RAG</th>";
        html += "<th>Modelo</th>";
        html += "<th>Costo</th>";
        html += "</tr></thead><tbody>";

        for (var i = 0; i < calls.length; i++) {
            var c = calls[i];
            html += "<tr>";
            html +=
                '<td><a class="call-link" href="#" data-call-id="' +
                escapeHTML(c.call_id) +
                '">' +
                escapeHTML(c.call_id) +
                "</a></td>";
            html += "<td>" + escapeHTML(c.patient_id) + "</td>";
            html += "<td>" + fmt(c.turn_count) + "</td>";
            html += "<td>" + fmtFloat(c.total_latency_ms, 1) + "</td>";
            html += "<td>" + fmt(c.total_input_tokens) + "</td>";
            html += "<td>" + fmt(c.total_output_tokens) + "</td>";
            html += "<td>" + fmt(c.total_rag_queries) + "</td>";
            html += "<td>" + fmt(c.model_calls) + "</td>";
            html += "<td>" + fmtCost(c.estimated_cost_usd) + "</td>";
            html += "</tr>";
        }

        html += "</tbody></table>";
        callsContent.innerHTML = html;

        // Attach click handlers to call-id links
        var links = callsContent.querySelectorAll(".call-link");
        for (var j = 0; j < links.length; j++) {
            links[j].addEventListener("click", function (e) {
                e.preventDefault();
                var cid = this.getAttribute("data-call-id");
                if (cid) {
                    loadCallDetail(cid);
                }
            });
        }
    }

    // -----------------------------------------------------------------------
    // Render: Call detail
    // -----------------------------------------------------------------------

    function renderCallDetail(data) {
        // Basic info header
        var html = '<div class="metrics-grid">';
        html += metricBox("Call ID", escapeHTML(data.call_id));
        html += metricBox("Paciente", escapeHTML(data.patient_id));
        html += metricBox("Turnos", fmt(data.turn_count));
        html += metricBox("Latencia Total (ms)", fmtFloat(data.total_latency_ms, 1));
        html += metricBox("Tokens Entrada", fmt(data.total_input_tokens));
        html += metricBox("Tokens Salida", fmt(data.total_output_tokens));
        html += metricBox("Consultas RAG", fmt(data.total_rag_queries));
        html += metricBox("Llamadas Modelo", fmt(data.model_calls));
        html += metricBox("Costo Estimado", fmtCost(data.estimated_cost_usd));
        html += "</div>";

        // Per-turn detail
        var turns = data.turns || [];
        html +=
            "<h3 style='margin-bottom:0.5rem;color:#334155'>Turnos (" +
            turns.length +
            ")</h3>";
        html += '<div class="turn-detail">';

        if (turns.length === 0) {
            html +=
                '<p class="placeholder-text">Sin turnos registrados.</p>';
        } else {
            for (var i = 0; i < turns.length; i++) {
                var t = turns[i];
                html += '<div class="turn-row">';
                html +=
                    '<div class="turn-header">Turno #' +
                    (t.turn_index + 1) +
                    " &mdash; " +
                    escapeHTML(t.timestamp) +
                    "</div>";
                html +=
                    '<div class="turn-field">Latencia total: ' +
                    fmtFloat(t.total_latency_ms, 1) +
                    " ms</div>";
                html +=
                    '<div class="turn-field">Modelo: ' +
                    escapeHTML(t.model) +
                    "</div>";
                html +=
                    '<div class="turn-field">Consultas RAG: ' +
                    fmt(t.rag_queries) +
                    "</div>";
                html +=
                    '<div class="turn-field">TTS: ' +
                    fmtFloat(t.tts_duration_ms, 1) +
                    " ms</div>";
                html +=
                    '<div class="turn-field">STT: ' +
                    fmtFloat(t.stt_duration_ms, 1) +
                    " ms</div>";
                html +=
                    '<div class="turn-field">LLM: ' +
                    fmtFloat(t.llm_duration_ms, 1) +
                    " ms</div>";
                html +=
                    '<div class="turn-field">Tokens entrada: ' +
                    fmt(t.input_tokens) +
                    "</div>";
                html +=
                    '<div class="turn-field">Tokens salida: ' +
                    fmt(t.output_tokens) +
                    "</div>";
                html +=
                    '<div class="turn-field">Costo estimado: ' +
                    fmtCost(t.estimated_cost_usd) +
                    "</div>";
                html += "</div>";
            }
        }
        html += "</div>";

        detailContent.innerHTML = html;
    }

    // -----------------------------------------------------------------------
    // Navigation
    // -----------------------------------------------------------------------

    function showList() {
        detailSection.classList.add("hidden");
        callsSection.classList.remove("hidden");
    }

    function showDetail(callId) {
        detailTitle.textContent = "Detalle de Llamada: " + callId;
        callsSection.classList.add("hidden");
        detailSection.classList.remove("hidden");
    }

    async function loadCallDetail(callId) {
        detailContent.innerHTML =
            '<p class="loading-text">Cargando detalles de ' +
            escapeHTML(callId) +
            "...</p>";
        showDetail(callId);
        try {
            var data = await fetchJSON("/metrics/calls/" + encodeURIComponent(callId));
            renderCallDetail(data);
        } catch (err) {
            detailContent.innerHTML =
                '<p class="error-message">Error al cargar detalles: ' +
                escapeHTML(String(err)) +
                "</p>";
        }
    }

    detailBackLink.addEventListener("click", function (e) {
        e.preventDefault();
        showList();
    });

    // -----------------------------------------------------------------------
    // Escape helper
    // -----------------------------------------------------------------------

    function escapeHTML(str) {
        var div = document.createElement("div");
        div.appendChild(document.createTextNode(str || ""));
        return div.innerHTML;
    }

    // -----------------------------------------------------------------------
    // Init
    // -----------------------------------------------------------------------

    async function init() {
        // Load summary
        try {
            var summary = await fetchJSON("/metrics/summary");
            renderSummary(summary);
        } catch (err) {
            summaryContent.innerHTML =
                '<p class="error-message">Error al cargar resumen: ' +
                escapeHTML(String(err)) +
                "</p>";
        }

        // Load calls list
        try {
            var calls = await fetchJSON("/metrics/calls");
            renderCallsList(calls);
        } catch (err) {
            callsContent.innerHTML =
                '<p class="error-message">Error al cargar llamadas: ' +
                escapeHTML(String(err)) +
                "</p>";
        }
    }

    init();
})();
