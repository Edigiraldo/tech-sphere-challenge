/**
 * app.js — Tech Sphere Challenge patient selection page.
 *
 * Handles the synthetic patient dropdown, patient-detail display, and the
 * start-call button that creates a call via POST /calls, stores the
 * response in sessionStorage, and navigates to the call interface.
 *
 * Depends on data.js (loaded via <script> in index.html) for the shared
 * PATIENTS catalogue.
 */

document.addEventListener("DOMContentLoaded", () => {

    // --- DOM references ----------------------------------------------------
    const patientSelect = document.getElementById("patient-select");
    const patientDetails = document.getElementById("patient-details");
    const detailId = document.getElementById("detail-id");
    const detailAge = document.getElementById("detail-age");
    const detailProcedure = document.getElementById("detail-procedure");
    const detailDay = document.getElementById("detail-day");
    const startCallBtn = document.getElementById("start-call-btn");
    const errorMessage = document.getElementById("error-message");
    const callStatusEl = document.getElementById("call-status");

    // --- Populate dropdown -------------------------------------------------
    PATIENTS.forEach((patient) => {
        const option = document.createElement("option");
        option.value = patient.id;
        option.textContent = `${patient.name} (${patient.id})`;
        patientSelect.appendChild(option);
    });

    // --- Show / hide patient details on selection --------------------------
    patientSelect.addEventListener("change", () => {
        const selectedId = patientSelect.value;
        errorMessage.classList.add("hidden");
        if (callStatusEl) callStatusEl.classList.add("hidden");

        if (!selectedId) {
            patientDetails.classList.add("hidden");
            startCallBtn.disabled = true;
            return;
        }

        const patient = PATIENTS.find((p) => p.id === selectedId);
        if (!patient) {
            patientDetails.classList.add("hidden");
            startCallBtn.disabled = true;
            errorMessage.textContent = "Paciente no encontrado.";
            errorMessage.classList.remove("hidden");
            return;
        }

        detailId.textContent = patient.id;
        detailAge.textContent = `${patient.age} años`;
        detailProcedure.textContent = patient.procedure;
        detailDay.textContent = `Día ${patient.postopDay}`;

        patientDetails.classList.remove("hidden");
        startCallBtn.disabled = false;
    });

    // --- Start call: POST /calls, store response, navigate -----------------
    startCallBtn.addEventListener("click", async () => {
        const selectedId = patientSelect.value;
        if (!selectedId) {
            errorMessage.textContent = "Seleccione un paciente antes de iniciar la llamada.";
            errorMessage.classList.remove("hidden");
            return;
        }

        const patient = PATIENTS.find((p) => p.id === selectedId);
        if (!patient) return;

        startCallBtn.disabled = true;
        errorMessage.classList.add("hidden");
        if (callStatusEl) {
            callStatusEl.textContent = "Creando llamada...";
            callStatusEl.classList.remove("hidden");
        }

        try {
            const response = await fetch("/calls", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    patient_id: patient.id,
                    dia_postop: patient.postopDay,
                    procedimiento: patient.procedure,
                    nombre_completo: patient.name,
                    eps: "EPS",
                }),
            });

            if (!response.ok) {
                let detail = "Error al crear la llamada.";
                try {
                    const err = await response.json();
                    detail = err.detail || detail;
                } catch (_) { /* use default */ }
                throw new Error(detail);
            }

            const data = await response.json();

            // Store call metadata for the call page via sessionStorage
            sessionStorage.setItem("callData", JSON.stringify({
                call_id: data.call_id,
                greeting_audio_b64: data.audio_base64,
                patient_id: patient.id,
                patient_name: patient.name,
                patient_procedure: patient.procedure,
                postop_day: patient.postopDay,
                total_questions: data.total_questions,
            }));

            window.location.href = "/call";
        } catch (err) {
            errorMessage.textContent = err.message;
            errorMessage.classList.remove("hidden");
            startCallBtn.disabled = false;
            if (callStatusEl) callStatusEl.classList.add("hidden");
        }
    });
});
