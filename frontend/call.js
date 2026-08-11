/**
 * call.js — Tech Sphere Challenge call interface.
 *
 * Real voice-call implementation:
 *   - Reads call_id and greeting audio from sessionStorage (set by app.js).
 *   - Uses MediaRecorder for microphone capture (audio/webm).
 *   - Sends base64-encoded audio to POST /calls/{call_id}/turn.
 *   - Plays agent responses as base64-decoded WAV via <audio>.
 *   - Displays transcription, conversation history, escalation alerts,
 *     and source citations.
 *   - Manages loading, error, and completed states.
 *
 * Depends on data.js (loaded via <script> in call.html) for the shared
 * PATIENTS catalogue.
 */

document.addEventListener("DOMContentLoaded", () => {

    // -----------------------------------------------------------------------
    // Call state
    // -----------------------------------------------------------------------
    const STATES = ["IDLE", "GREETING", "CONSENT", "QUESTIONS", "CLOSING", "ENDED"];
    const STATE_CLASSES = [
        "state-idle", "state-greeting", "state-consent",
        "state-questions", "state-closing", "state-ended",
    ];

    let currentState = "IDLE";
    let isRecording = false;
    let callStartTime = null;
    let timerInterval = null;
    let callEnded = false;
    let callId = null;
    let totalQuestions = 6;

    // MediaRecorder
    let mediaRecorder = null;
    let audioChunks = [];

    const messages = [];

    // -----------------------------------------------------------------------
    // DOM references
    // -----------------------------------------------------------------------
    const callStateBadge = document.getElementById("call-state-badge");
    const patientNameDisplay = document.getElementById("patient-name-display");
    const transcriptArea = document.getElementById("transcript-area");
    const conversationHistory = document.getElementById("conversation-history");
    const recordToggleBtn = document.getElementById("record-toggle-btn");
    const recordIcon = document.getElementById("record-icon");
    const recordLabel = document.getElementById("record-label");
    const endCallBtn = document.getElementById("end-call-btn");
    const callTimerEl = document.getElementById("call-timer");
    const audioPlayer = document.getElementById("audio-player");
    const audioStatus = document.getElementById("audio-status");
    const escalationBanner = document.getElementById("escalation-banner");
    const escalationText = document.getElementById("escalation-text");
    const loadingOverlay = document.getElementById("loading-overlay");
    const loadingText = document.getElementById("loading-text");
    const callCompletedBanner = document.getElementById("call-completed-banner");
    const inlineSummarySection = document.getElementById("inline-summary-section");
    const inlineSummaryContent = document.getElementById("inline-summary-content");
    const viewFullSummaryLink = document.getElementById("view-full-summary-link");

    // -----------------------------------------------------------------------
    // Read call data from sessionStorage
    // -----------------------------------------------------------------------
    let callData = null;
    try {
        const raw = sessionStorage.getItem("callData");
        if (raw) {
            callData = JSON.parse(raw);
        }
    } catch (_) { /* malformed — treat as missing */ }

    callId = callData ? callData.call_id : null;
    totalQuestions = callData ? (callData.total_questions || 6) : 6;

    /**
     * Display patient name in the header.
     */
    function initPatientDisplay() {
        if (callData) {
            patientNameDisplay.textContent =
                `${callData.patient_name} — ${callData.patient_procedure}`;
        } else {
            const urlParams = new URLSearchParams(window.location.search);
            const patientId = urlParams.get("patient_id");
            const patient = PATIENTS.find((p) => p.id === patientId) || null;
            if (patient) {
                patientNameDisplay.textContent = `${patient.name} — ${patient.procedure}`;
            } else {
                patientNameDisplay.textContent = patientId
                    ? `Paciente ${patientId}`
                    : "Paciente no especificado";
            }
        }
    }

    // -----------------------------------------------------------------------
    // UI state helpers
    // -----------------------------------------------------------------------

    /** Show / hide the loading overlay. */
    function setLoading(visible, text) {
        if (loadingOverlay) {
            loadingOverlay.classList.toggle("hidden", !visible);
        }
        if (loadingText && text) {
            loadingText.textContent = text;
        }
    }

    /** Show the completed banner. */
    function showCompleted() {
        if (callCompletedBanner) {
            callCompletedBanner.classList.remove("hidden");
        }
        setLoading(false);
        // Fetch and render the inline summary once the call completes.
        fetchInlineSummary();
    }

    /** Fetch the call summary from the API and render it inline. */
    async function fetchInlineSummary() {
        if (!callId || !inlineSummarySection || !inlineSummaryContent) return;

        try {
            const response = await fetch(`/calls/${encodeURIComponent(callId)}/summary`);

            if (!response.ok) {
                // Summary may not be available yet — retry once after a short delay.
                setTimeout(async () => {
                    try {
                        const retryResp = await fetch(`/calls/${encodeURIComponent(callId)}/summary`);
                        if (retryResp.ok) {
                            const data = await retryResp.json();
                            renderInlineSummary(data);
                        }
                    } catch (_) { /* silently skip */ }
                }, 800);
                return;
            }

            const data = await response.json();
            renderInlineSummary(data);
        } catch (_) {
            // Summary fetch is best-effort on the call page.
        }
    }

    /**
     * Render a compact summary inline on the call page.
     * @param {object} data - SummaryResponse from the API
     */
    function renderInlineSummary(data) {
        if (!inlineSummarySection || !inlineSummaryContent) return;

        // Update the "View full summary" link
        if (viewFullSummaryLink) {
            viewFullSummaryLink.href = `/summary?call_id=${encodeURIComponent(data.call_id)}`;
        }

        let html = "";

        // Patient + procedure (compact)
        html += '<div class="summary-section">';
        html += `<p class="summary-section-content">${escapeHtml(data.patient_summary)}</p>`;
        html += '</div>';

        html += '<div class="summary-section">';
        html += `<p class="summary-section-content">${escapeHtml(data.procedure_summary)}</p>`;
        html += '</div>';

        // Decision with severity colouring
        const sevClass = getInlineSeverityClass(data.decision_summary);
        html += '<div class="summary-section' + (sevClass ? ' ' + sevClass : '') + '">';
        html += '<h3 class="summary-section-heading">&#x26A0; Decisión</h3>';
        html += `<p class="summary-section-content">${escapeHtml(data.decision_summary)}</p>`;
        html += '</div>';

        // Next steps
        html += '<div class="summary-section">';
        html += '<h3 class="summary-section-heading">&#x27A1; Próximos Pasos</h3>';
        html += `<p class="summary-section-content">${escapeHtml(data.next_steps)}</p>`;
        html += '</div>';

        // Sources count
        if (data.sources && data.sources.length > 0) {
            html += '<div class="summary-section">';
            html += `<p class="summary-section-content">${data.sources.length} fuente(s) citada(s) — disponible en el resumen completo.</p>`;
            html += '</div>';
        }

        inlineSummaryContent.innerHTML = html;
        inlineSummarySection.classList.remove("hidden");
    }

    /**
     * Derive a severity CSS class from decision text for inline rendering.
     * @param {string} text
     * @returns {string}
     */
    function getInlineSeverityClass(text) {
        if (!text) return "";
        const lower = text.toLowerCase();
        if (lower.includes("rojo") || lower.includes("inmediato")) return "summary-severity-red";
        if (lower.includes("amarillo") || lower.includes("precaucion") || lower.includes("precaución")) return "summary-severity-yellow";
        if (lower.includes("verde") || lower.includes("normal")) return "summary-severity-green";
        return "";
    }

    /** Show an error in the transcript area and disable controls. */
    function showError(message) {
        setLoading(false);
        setCallState("ENDED");
        addMessage("system", `⚠️ Error: ${message}`);
        recordToggleBtn.disabled = true;
        endCallBtn.disabled = true;
    }

    // -----------------------------------------------------------------------
    // State display
    // -----------------------------------------------------------------------

    function setCallState(newState) {
        if (!STATES.includes(newState)) {
            console.warn("Invalid state:", newState);
            return;
        }
        currentState = newState;
        STATE_CLASSES.forEach((cls) => callStateBadge.classList.remove(cls));
        callStateBadge.classList.add(`state-${newState.toLowerCase()}`);
        callStateBadge.textContent = newState;

        if (newState === "ENDED") {
            stopRecording();
            endCallBtn.disabled = true;
            recordToggleBtn.disabled = true;
        }
    }

    // -----------------------------------------------------------------------
    // Escalation banner
    // -----------------------------------------------------------------------

    /**
     * Show or update the escalation banner.
     * @param {object|null} escalation - EscalationInfo from API or null
     */
    function showEscalation(escalation) {
        if (!escalationBanner || !escalationText) return;

        if (!escalation) {
            escalationBanner.classList.add("hidden");
            return;
        }

        // Remove old severity classes
        escalationBanner.classList.remove("esc-green", "esc-yellow", "esc-red");
        escalationBanner.classList.add(`esc-${escalation.severity.toLowerCase()}`);
        escalationBanner.classList.remove("hidden");

        const domainLabel = escalation.domain || "general";
        escalationText.textContent =
            `[${escalation.severity}] ${domainLabel}: ${escalation.reason}`;
    }

    // -----------------------------------------------------------------------
    // Conversation history
    // -----------------------------------------------------------------------

    function addMessage(role, text, citations) {
        messages.push({ role, text, citations: citations || [], timestamp: new Date() });
        renderMessages();
    }

    function renderMessages() {
        if (messages.length === 0) {
            conversationHistory.innerHTML =
                '<p class="placeholder-text">El historial de la conversación aparecerá aquí.</p>';
            return;
        }

        conversationHistory.innerHTML = "";
        messages.forEach((msg) => {
            const msgDiv = document.createElement("div");
            msgDiv.classList.add("message", `message-${msg.role}`);

            const roleLabels = {
                agent: "Agente",
                patient: "Paciente",
                system: "Sistema",
            };

            let html =
                `<span class="role-label">${roleLabels[msg.role]}</span>` +
                `<span class="message-text">${escapeHtml(msg.text)}</span>`;

            // Render citations when present
            if (msg.citations && msg.citations.length > 0) {
                html += '<div class="citations-list">';
                html += '<span class="citations-label">Fuentes:</span>';
                msg.citations.forEach((cit) => {
                    html += `<span class="citation-badge" title="${escapeHtml(cit.source_filename || '')} p.${cit.page_number || '?'}">${escapeHtml(cit.source_filename || cit.document_id || '—')}</span>`;
                });
                html += '</div>';
            }

            msgDiv.innerHTML = html;
            conversationHistory.appendChild(msgDiv);
        });

        conversationHistory.scrollTop = conversationHistory.scrollHeight;
    }

    // -----------------------------------------------------------------------
    // Transcript
    // -----------------------------------------------------------------------

    function setTranscript(text) {
        transcriptArea.innerHTML = "";
        const p = document.createElement("p");
        p.textContent = text;
        transcriptArea.appendChild(p);
    }

    // -----------------------------------------------------------------------
    // Audio playback (base64 WAV)
    // -----------------------------------------------------------------------

    /**
     * Decode a base64 WAV string and play it through the audio element.
     * @param {string} base64Data
     * @returns {Promise<void>} Resolves when playback starts.
     */
    function playBase64Audio(base64Data) {
        return new Promise((resolve, reject) => {
            try {
                const binary = atob(base64Data);
                const bytes = new Uint8Array(binary.length);
                for (let i = 0; i < binary.length; i++) {
                    bytes[i] = binary.charCodeAt(i);
                }
                const blob = new Blob([bytes], { type: "audio/wav" });
                const url = URL.createObjectURL(blob);

                // Revoke previous object URL
                if (audioPlayer.dataset.blobUrl) {
                    URL.revokeObjectURL(audioPlayer.dataset.blobUrl);
                }
                audioPlayer.dataset.blobUrl = url;

                audioPlayer.src = url;
                audioPlayer.style.display = "block";
                audioStatus.textContent = "Reproduciendo respuesta del agente...";

                audioPlayer.onended = () => {
                    audioStatus.textContent = "Audio reproducido";
                    resolve();
                };
                audioPlayer.onerror = () => {
                    audioStatus.textContent = "Error al reproducir audio";
                    reject(new Error("Audio playback failed"));
                };

                audioPlayer.play().catch((err) => {
                    audioStatus.textContent = "Error al iniciar reproducción";
                    reject(err);
                });
            } catch (err) {
                audioStatus.textContent = "Error al decodificar audio";
                reject(err);
            }
        });
    }

    // -----------------------------------------------------------------------
    // Recording (MediaRecorder)
    // -----------------------------------------------------------------------

    async function startRecording() {
        if (isRecording) return;

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            // Prefer webm/opus; fall back to whatever is available
            let mimeType = "audio/webm";
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                mimeType = "audio/webm;codecs=opus";
            }
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                // Let the browser choose
                mimeType = "";
            }

            const options = mimeType ? { mimeType } : {};
            mediaRecorder = new MediaRecorder(stream, options);
            audioChunks = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                // Stop all tracks in the stream
                if (mediaRecorder.stream) {
                    mediaRecorder.stream.getTracks().forEach((track) => track.stop());
                }

                if (audioChunks.length === 0) return;

                const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
                const reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                reader.onloadend = async () => {
                    // Extract base64 payload after "data:...;base64,"
                    const result = reader.result;
                    const commaIdx = result.indexOf(",");
                    const base64 = commaIdx >= 0 ? result.substring(commaIdx + 1) : result;
                    setLoading(true, "Procesando respuesta...");
                    try {
                        await sendTurn(base64);
                    } catch (err) {
                        showError(err.message);
                    }
                };
            };

            mediaRecorder.start();
            isRecording = true;
            updateRecordingUI();
            setTranscript("Grabando... hable ahora.");
        } catch (err) {
            showError("No se pudo acceder al micrófono. Verifique los permisos.");
            console.error("getUserMedia error:", err);
        }
    }

    function stopRecording() {
        if (!isRecording || !mediaRecorder) return;
        if (mediaRecorder.state === "recording") {
            mediaRecorder.stop();
        }
        isRecording = false;
        updateRecordingUI();
    }

    function updateRecordingUI() {
        if (isRecording) {
            recordToggleBtn.classList.add("recording");
            recordIcon.innerHTML = "&#x23F9;"; // stop square
            recordLabel.textContent = "Detener Grabación";
        } else {
            recordToggleBtn.classList.remove("recording");
            recordIcon.innerHTML = "&#x23FA;"; // record circle
            recordLabel.textContent = "Iniciar Grabación";
        }
    }

    // -----------------------------------------------------------------------
    // API calls
    // -----------------------------------------------------------------------

    /**
     * Send a voice turn to the backend.
     * @param {string} audioBase64 - Base64-encoded audio data
     */
    async function sendTurn(audioBase64) {
        if (callEnded || !callId) return;

        const response = await fetch(`/calls/${callId}/turn`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ audio_base64: audioBase64 }),
        });

        if (!response.ok) {
            let detail = "Error en el turno de voz.";
            try {
                const err = await response.json();
                detail = err.detail || detail;
            } catch (_) { /* use default */ }
            throw new Error(detail);
        }

        const data = await response.json();
        handleTurnResponse(data);
    }

    /**
     * Process a TurnResponse from the backend.
     * @param {object} data - Parsed JSON TurnResponse
     */
    async function handleTurnResponse(data) {
        setLoading(false);

        // Update state
        if (data.state) {
            setCallState(data.state);
        }

        // Add patient transcription as a patient message.
        // When the backend returns a patient_transcription field we use it;
        // otherwise we show a placeholder (backward-compatible with older
        // backend versions that did not include this field).
        addMessage(
            "patient",
            data.patient_transcription || "🎤 [grabación enviada]"
        );

        // Add agent message
        if (data.transcription) {
            addMessage("agent", data.transcription, data.citations || []);
        }

        // Display escalation info
        showEscalation(data.escalation || null);

        // Check if call ended
        if (data.call_ended) {
            callEnded = true;
            stopTimer();
            stopRecording();
            recordToggleBtn.disabled = true;
            endCallBtn.disabled = true;
            showCompleted();
            addMessage("system", "Llamada finalizada.");
            setTranscript("Llamada completada.");
            return;
        }

        // Play agent audio response (if present)
        if (data.audio_base64) {
            setLoading(true, "Reproduciendo respuesta...");
            try {
                await playBase64Audio(data.audio_base64);
            } catch (err) {
                console.error("Audio playback error:", err);
            }
            setLoading(false);
        }

        // Enable recording for next turn if response is required
        if (data.requires_response && !data.call_ended) {
            recordToggleBtn.disabled = false;
            setTranscript("Listo para responder. Presione Iniciar Grabación.");
            audioStatus.textContent = data.state === "CONSENT"
                ? "El agente está esperando su consentimiento."
                : "El agente está esperando su respuesta.";
        }
    }

    // -----------------------------------------------------------------------
    // Call timer
    // -----------------------------------------------------------------------

    function startTimer() {
        callStartTime = Date.now();
        updateTimerDisplay();
        timerInterval = setInterval(updateTimerDisplay, 1000);
    }

    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
        updateTimerDisplay();
    }

    function updateTimerDisplay() {
        if (!callStartTime) {
            callTimerEl.textContent = "00:00";
            return;
        }
        const elapsed = Math.floor((Date.now() - callStartTime) / 1000);
        const minutes = String(Math.floor(elapsed / 60)).padStart(2, "0");
        const seconds = String(elapsed % 60).padStart(2, "0");
        callTimerEl.textContent = `${minutes}:${seconds}`;
    }

    // -----------------------------------------------------------------------
    // Call lifecycle
    // -----------------------------------------------------------------------

    /** Start the call: play greeting audio, then enable recording. */
    async function startCall() {
        if (!callData) {
            showError("Datos de llamada no encontrados. Regrese a la página de selección.");
            return;
        }

        setLoading(true, "Iniciando llamada...");
        startTimer();
        endCallBtn.disabled = false;

        // Set initial state
        setCallState("GREETING");

        // Play the greeting audio that was returned by POST /calls
        if (callData.greeting_audio_b64) {
            addMessage("system", "Llamada iniciada.");
            setTranscript("Reproduciendo saludo del agente...");
            try {
                await playBase64Audio(callData.greeting_audio_b64);
            } catch (err) {
                console.error("Greeting playback error:", err);
            }
        }

        setLoading(false);
        addMessage("agent", "El agente le está hablando. Por favor escuche y responda.");

        // Enable recording
        recordToggleBtn.disabled = false;
        setTranscript("Presione Iniciar Grabación para responder.");
        audioStatus.textContent = "Micrófono listo. Presione el botón para hablar.";
    }

    /** End the call (user-initiated). */
    function endCall() {
        stopRecording();
        stopTimer();
        callEnded = true;
        setCallState("ENDED");
        recordToggleBtn.disabled = true;
        endCallBtn.disabled = true;
        addMessage("system", "Llamada finalizada por el usuario.");
        setTranscript("Llamada finalizada.");
        showCompleted();
    }

    // -----------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }

    // -----------------------------------------------------------------------
    // Recording toggle (event handler)
    // -----------------------------------------------------------------------
    async function toggleRecording() {
        if (callEnded) return;

        if (isRecording) {
            // Stop recording → triggers onstop → sends turn
            stopRecording();
            recordToggleBtn.disabled = true;
            setTranscript("Procesando audio...");
        } else {
            // Start recording
            await startRecording();
        }
    }

    // -----------------------------------------------------------------------
    // Event listeners
    // -----------------------------------------------------------------------
    recordToggleBtn.addEventListener("click", toggleRecording);
    endCallBtn.addEventListener("click", endCall);

    // -----------------------------------------------------------------------
    // Initialisation
    // -----------------------------------------------------------------------
    initPatientDisplay();
    setCallState("IDLE");

    if (callData) {
        // Brief delay so the user sees the IDLE state
        setTimeout(startCall, 300);
    } else {
        setCallState("IDLE");
        setTranscript("No se encontraron datos de llamada. Regrese a la página principal.");
        recordToggleBtn.disabled = true;
    }
});
