/**
 * data.js — Tech Sphere Challenge shared synthetic patient catalogue.
 *
 * Loaded before app.js and call.js so both pages share a single source of
 * truth for the local-only synthetic patient data.
 */
const PATIENTS = [
    {
        id: "P001",
        name: "Paciente 001",
        age: 45,
        procedure: "Apendicectomía laparoscópica",
        postopDay: 3,
    },
    {
        id: "P002",
        name: "Paciente 002",
        age: 62,
        procedure: "Colecistectomía",
        postopDay: 5,
    },
    {
        id: "P003",
        name: "Paciente 003",
        age: 38,
        procedure: "Hernioplastia inguinal",
        postopDay: 2,
    },
    {
        id: "P004",
        name: "Paciente 004",
        age: 55,
        procedure: "Cesárea",
        postopDay: 4,
    },
    {
        id: "P005",
        name: "Paciente 005",
        age: 71,
        procedure: "Reemplazo total de cadera",
        postopDay: 7,
    },
];
