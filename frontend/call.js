/**
 * call.js — Tech Sphere Challenge call interface.
 *
 * Manages the call UI: state display, record/stop placeholder toggle,
 * conversation history rendering, transcript area, audio player placeholder,
 * and call timer.  All behaviour is local; no microphone, WebSocket, or
 * backend API is used.  This is a UI shell for local-only operation.
 *
 * Depends on data.js (loaded via <script> in call.html) for the shared
 * PATIENTS catalogue.
 */

document.addEventListener("DOMContentLoaded", () => {

    // -----------------------------------------------------------------------
    // Call state machine (local-only, mirrors backend/domain states)
    // -----------------------------------------------------------------------
    const STATES = ["IDLE", "GREETING", "CONSENT", "QUESTIONS", "CLOSING", "ENDED"];
    let currentState = "IDLE";

    /** All CSS classes valid on the state-badge element. */
    const STATE_CLASSES = [
        "state-idle", "state-greeting", "state-consent",
        "state-questions", "state-closing", "state-ended",
    ];

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

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    let isRecording = false;
    let callStartTime = null;
    let timerInterval = null;
    const messages = [];

    // -----------------------------------------------------------------------
    // Read patient_id from query string
    // -----------------------------------------------------------------------
    const urlParams = new URLSearchParams(window.location.search);
    const patientId = urlParams.get("patient_id");

    const patient = PATIENTS.find((p) => p.id === patientId) || null;

    /**
     * Display patient name (or fallback) in the header.
     */
    function initPatientDisplay() {
        if (patient) {
            patientNameDisplay.textContent = `${patient.name} — ${patient.procedure}`;
        } else {
            patientNameDisplay.textContent = patientId
                ? `Paciente ${patientId}`
                : "Paciente no especificado";
        }
    }

    // -----------------------------------------------------------------------
    // State display
    // -----------------------------------------------------------------------

    /**
     * Transition to a new call state and update the badge.
     * @param {string} newState - One of the valid STATES.
     */
    function setCallState(newState) {
        if (!STATES.includes(newState)) {
            console.warn("Invalid state:", newState);
            return;
        }
        currentState = newState;

        // Remove all known state classes, then add the correct one
        STATE_CLASSES.forEach((cls) => callStateBadge.classList.remove(cls));
        callStateBadge.classList.add(`state-${newState.toLowerCase()}`);
        callStateBadge.textContent = newState;

        if (newState === "ENDED") {
            stopRecording();
            endCallBtn.disabled = true;
        }
    }

    // -----------------------------------------------------------------------
    // Conversation history
    // -----------------------------------------------------------------------

    /**
     * Add a message to the conversation history and re-render.
     * @param {"agent"|"patient"|"system"} role
     * @param {string} text
     */
    function addMessage(role, text) {
        messages.push({ role, text, timestamp: new Date() });
        renderMessages();
    }

    /** Render all messages in the conversation-history div. */
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

            msgDiv.innerHTML =
                `<span class="role-label">${roleLabels[msg.role]}</span>` +
                `<span class="message-text">${escapeHtml(msg.text)}</span>`;

            conversationHistory.appendChild(msgDiv);
        });

        // Auto-scroll to bottom
        conversationHistory.scrollTop = conversationHistory.scrollHeight;
    }

    // -----------------------------------------------------------------------
    // Transcript
    // -----------------------------------------------------------------------

    /**
     * Update the transcript area with the given text.
     * @param {string} text
     */
    function setTranscript(text) {
        transcriptArea.innerHTML = "";
        const p = document.createElement("p");
        p.textContent = text;
        transcriptArea.appendChild(p);
    }

    // -----------------------------------------------------------------------
    // Audio placeholder
    // -----------------------------------------------------------------------

    /** Show a placeholder audio URL (no real audio data). */
    function setAudioPlaceholder() {
        audioStatus.textContent = "Audio de respuesta (simulación)";
        audioPlayer.style.display = "none";
    }

    // -----------------------------------------------------------------------
    // Recording toggle
    // -----------------------------------------------------------------------

    /** Toggle the recording state. */
    function toggleRecording() {
        isRecording = !isRecording;

        if (isRecording) {
            recordToggleBtn.classList.add("recording");
            recordIcon.innerHTML = "&#x23F9;"; // stop square
            recordLabel.textContent = "Detener Grabación";
            setTranscript("Grabando... (simulación — no se captura audio real)");
        } else {
            recordToggleBtn.classList.remove("recording");
            recordIcon.innerHTML = "&#x23FA;"; // record circle
            recordLabel.textContent = "Iniciar Grabación";
            setTranscript("Grabación detenida (simulación)");
        }
    }

    /** Force-stop recording (used on call end). */
    function stopRecording() {
        if (!isRecording) return;
        isRecording = false;
        recordToggleBtn.classList.remove("recording");
        recordIcon.innerHTML = "&#x23FA;";
        recordLabel.textContent = "Iniciar Grabación";
        recordToggleBtn.disabled = true;
    }

    // -----------------------------------------------------------------------
    // Call timer
    // -----------------------------------------------------------------------

    /** Start the call timer. */
    function startTimer() {
        callStartTime = Date.now();
        updateTimerDisplay();
        timerInterval = setInterval(updateTimerDisplay, 1000);
    }

    /** Stop the call timer. */
    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
        updateTimerDisplay();
    }

    /** Update the MM:SS display. */
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

    /** Start the call flow (local simulation). */
    function startCall() {
        setCallState("GREETING");
        startTimer();
        recordToggleBtn.disabled = false;
        endCallBtn.disabled = false;

        // Simulate initial agent greeting
        addMessage("system", "Llamada iniciada.");
        addMessage("agent", "Buenos días. ¿Cómo se siente el día de hoy?");

        setAudioPlaceholder();
    }

    /** End the call. */
    function endCall() {
        stopRecording();
        stopTimer();
        setCallState("ENDED");
        addMessage("system", "Llamada finalizada.");
    }

    // -----------------------------------------------------------------------
    // Utility
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

    // Auto-start call flow after a brief delay so the user sees the IDLE state
    setTimeout(startCall, 300);
});
