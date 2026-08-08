/**
 * app.js — Tech Sphere Challenge patient selection page.
 *
 * Handles the synthetic patient dropdown, patient-detail display, and the
 * start-call button that navigates to the call interface.
 *
 * All data is local/static (no backend API calls).  This is a UI shell
 * designed for local-only operation.
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

    // --- Start call: navigate to call page with patient id -----------------
    startCallBtn.addEventListener("click", () => {
        const selectedId = patientSelect.value;
        if (!selectedId) {
            errorMessage.textContent = "Seleccione un paciente antes de iniciar la llamada.";
            errorMessage.classList.remove("hidden");
            return;
        }
        window.location.href = `/call?patient_id=${encodeURIComponent(selectedId)}`;
    });
});
