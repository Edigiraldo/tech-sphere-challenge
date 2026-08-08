"""Spanish postoperative symptom lexicon for the escalation engine.

This module is the **text-only, stdlib-only knowledge base** that the
deterministic rule engine in ``rules.py`` consults.  It defines:

* **Red-flag keywords** per symptom domain — any match → RED.
* **Yellow triggers** per domain — concerning but not immediately critical.
* **Green indicators** per domain — reassuring patterns.
* **Negation markers** that invert keyword match polarity.
* **Numeric thresholds** for pain scores and temperature values.
* **Ambiguity phrases** that indicate unclear answers.
* **Cross-cutting red flags** that escalate regardless of domain.

All keywords are in **lowercase Spanish**.  The rule engine lowercases
patient input before matching.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Domain names (used as keys in the categorised lexicons below)
# ---------------------------------------------------------------------------

DOMAIN_DOLOR = "dolor"
DOMAIN_FIEBRE = "fiebre"
DOMAIN_HERIDA = "herida"
DOMAIN_APETITO = "apetito"
DOMAIN_SUENO = "sueno"
DOMAIN_MOVILIDAD = "movilidad"

ALL_DOMAINS: tuple[str, ...] = (
    DOMAIN_DOLOR,
    DOMAIN_FIEBRE,
    DOMAIN_HERIDA,
    DOMAIN_APETITO,
    DOMAIN_SUENO,
    DOMAIN_MOVILIDAD,
)

# ---------------------------------------------------------------------------
# Red-flag keyword lists (any match → immediate RED)
# ---------------------------------------------------------------------------

RED_FLAGS: dict[str, list[str]] = {
    DOMAIN_DOLOR: [
        # Intensity (implicit >= 8 or described as unbearable)
        "insoportable",
        "no aguanto",
        "peor dolor",
        "dolor intenso",
        "dolor severo",
        "dolor muy fuerte",
        "no soporto",
        "me muero",
        "terrible",
        "horrible",
        "atroz",
        "no puedo mas",
        "no resisto",
        "grito",
        "lloro",
        "no me deja",
        "incapacitante",
        "no me puedo mover del dolor",
        "el peor dolor de mi vida",
        "emergencia",
        "urgencia",
        "me desmayo del dolor",
        "dolor que no cede",
        "no me calma nada",
        "no mejora con nada",
    ],
    DOMAIN_FIEBRE: [
        "fiebre alta",
        "fiebre muy alta",
        "fiebre de 40",
        "fiebre de 39",
        "temperatura muy alta",
        "no baja la fiebre",
        "no cede la fiebre",
        "fiebre que no cede",
        "escalofríos con fiebre",
        "convulsiones",
        "convulsion",
        "delirio",
        "alucinaciones",
        "confusión",
        "confundido",
        "no responde",
        "inconsciente",
        "me desmayé",
        "me desmayo",
        "sudoración profusa",
        "taquicardia",
        "corazón acelerado",
        "dificultad para respirar",
        "no puedo respirar",
        "falta de aire con fiebre",
    ],
    DOMAIN_HERIDA: [
        # Dehiscence (opening)
        "se abrió",
        "se ha abierto",
        "abierta",
        "se abrio",
        "abrio",
        "puntos sueltos",
        "sutura abierta",
        # Purulent secretion
        "pus",
        "supuración",
        "supurando",
        "secreción purulenta",
        "secreción amarilla",
        "secreción verde",
        "secrecion",
        "líquido maloliente",
        "liquido maloliente",
        # Foul odor
        "mal olor",
        "huele mal",
        "hedor",
        "fétida",
        "fétido",
        "hedionda",
        # Spreading redness / cellulitis
        "enrojecimiento que se extiende",
        "rojo y caliente",
        "caliente al tacto",
        "línea roja",
        "raya roja",
        # Necrosis
        "negra",
        "necrosis",
        "tejido muerto",
        "gangrena",
        # Bleeding
        "sangrado abundante",
        "hemorragia",
        "sangrando mucho",
        "no para de sangrar",
        "sangre",
    ],
    DOMAIN_APETITO: [
        "no como",
        "no he comido",
        "no puedo comer",
        "no tolero nada",
        "no tolero",
        "vómito todo",
        "vomito todo",
        "no me pasa nada",
        "no retengo",
        "no retengo nada",
        "deshidratado",
        "deshidratada",
        "no bebo agua",
        "no he bebido",
        "no tomo líquidos",
        "no tomo liquidos",
        "vómito con sangre",
        "vomito con sangre",
        "heces negras",
        "sangre en heces",
        "dolor abdominal intenso",
        "no orino",
        "no he orinado",
    ],
    DOMAIN_SUENO: [
        # Severe sleep deprivation + pain (may indicate serious issue)
        "no duermo nada",
        "no he dormido nada",
        "insomnio total",
        "no puedo dormir del dolor",
        "me despierta el dolor",
        "me despierto gritando",
        "pesadillas constantes",
        "no descanso nada",
        "alucinaciones por falta de sueño",
        "delirando por falta de sueño",
    ],
    DOMAIN_MOVILIDAD: [
        "no me puedo levantar",
        "no me puedo mover",
        "no puedo caminar",
        "postrado",
        "postrada",
        "no me sostengo",
        "me caí",
        "me cai",
        "caída",
        "caida",
        "me caigo",
        "pérdida de fuerza total",
        "perdida de fuerza total",
        "parálisis",
        "paralisis",
        "no siento las piernas",
        "no siento los brazos",
        "dolor en el pecho",
        "opresión en el pecho",
        "opresion en el pecho",
        "falta de aire",
        "me ahogo",
        "no puedo respirar",
        "dificultad para respirar",
        "disnea",
        "mareo intenso",
        "me desmayo al levantarme",
        "pérdida del conocimiento",
        "perdida del conocimiento",
        "visión borrosa",
        "vision borrosa",
    ],
}

# ---------------------------------------------------------------------------
# Yellow trigger keyword lists (concerning but not immediately RED)
# ---------------------------------------------------------------------------

YELLOW_TRIGGERS: dict[str, list[str]] = {
    DOMAIN_DOLOR: [
        "dolor moderado",
        "me duele bastante",
        "no me ha bajado",
        "sigue igual",
        "igual de fuerte",
        "molestia",
        "incomodidad",
        "me molesta",
        "no mejora",
        "empeorando",
        "está peor",
        "esta peor",
    ],
    DOMAIN_FIEBRE: [
        "fiebre",
        "calentura",
        "escalofríos",
        "escalofrios",
        "fiebrecita",
        "poquito de fiebre",
        "febrícula",
        "febricula",
        "destemplado",
        "destemplada",
    ],
    DOMAIN_HERIDA: [
        "enrojecimiento",
        "roja",
        "rojo",
        "hinchazón",
        "hinchazon",
        "hinchada",
        "hinchado",
        "inflamada",
        "inflamado",
        "me pica",
        "picazón",
        "picazon",
        "comezón",
        "comezon",
        "calor",
        "caliente",
        "sensibilidad",
        "sensible",
        "mojada",
        "húmeda",
        "humeda",
        "supura un poquito",
        "un poco de líquido",
        "un poco de liquido",
    ],
    DOMAIN_APETITO: [
        "poco apetito",
        "no tengo hambre",
        "no me da hambre",
        "casi no como",
        "me cuesta comer",
        "náuseas",
        "nauseas",
        "asco",
        "vómito",
        "vomito",
        "he vomitado",
        "malestar estomacal",
        "no me sienta bien",
        "pesadez",
        "lleno",
        "llena",
        "sin ganas",
    ],
    DOMAIN_SUENO: [
        "duermo mal",
        "duermo poco",
        "me cuesta dormir",
        "me despierto",
        "me despierto mucho",
        "sueño ligero",
        "sueño interrumpido",
        "insomnio",
        "no descanso bien",
        "cansado",
        "cansada",
        "fatiga",
        "agotado",
        "agotada",
    ],
    DOMAIN_MOVILIDAD: [
        "me cuesta",
        "dificultad",
        "apenas",
        "con ayuda",
        "necesito ayuda",
        "débil",
        "debil",
        "debilidad",
        "mareo",
        "mareada",
        "mareado",
        "me tambaleo",
        "inestable",
        "no tengo fuerza",
        "me canso",
        "me agito",
        "agitado",
        "agitada",
        "lento",
        "lenta",
        "me duele al moverme",
        "no puedo mucho",
        "solo un poco",
        "poquito",
    ],
}

# ---------------------------------------------------------------------------
# Green indicator keyword lists (reassuring patterns)
# ---------------------------------------------------------------------------

GREEN_INDICATORS: dict[str, list[str]] = {
    DOMAIN_DOLOR: [
        "bien",
        "mejor",
        "mejorando",
        "poquito",
        "leve",
        "suave",
        "casi nada",
        "no tengo dolor",
        "sin dolor",
        "no me duele",
        "cero",
        "nada",
        "tranquilo",
        "muy bien",
        "estoy bien",
        "controlado",
        "desapareció",
        "desaparecio",
        "se fue",
        "ha bajado",
        "menos",
        "disminuyendo",
    ],
    DOMAIN_FIEBRE: [
        "no tengo fiebre",
        "sin fiebre",
        "temperatura normal",
        "normal",
        "no he tenido",
        "ninguna",
        "nada de fiebre",
        "no me ha dado",
        "bien",
        "todo bien",
        "no ha habido",
    ],
    DOMAIN_HERIDA: [
        "bien",
        "normal",
        "cicatrizando",
        "cerrada",
        "seca",
        "seco",
        "limpia",
        "limpio",
        "sin problema",
        "sin problemas",
        "sin molestia",
        "sin molestias",
        "todo bien",
        "sana",
        "sano",
        "sanando",
        "curando",
        "bien cerrada",
        "no hay",
        "nada",
        "ningún problema",
        "ningun problema",
    ],
    DOMAIN_APETITO: [
        "bien",
        "normal",
        "buen apetito",
        "como bien",
        "estoy comiendo",
        "he comido",
        "sin problema",
        "sin problemas",
        "todo bien",
        "mejor",
        "mejorando",
        "bebo bien",
        "líquidos bien",
        "liquidos bien",
    ],
    DOMAIN_SUENO: [
        "bien",
        "duermo bien",
        "he dormido bien",
        "descanso bien",
        "descansando",
        "normal",
        "sin problema",
        "sin problemas",
        "tranquilo",
        "tranquila",
        "profundo",
        "toda la noche",
        "mejor",
        "mejorando",
    ],
    DOMAIN_MOVILIDAD: [
        "bien",
        "normal",
        "camino bien",
        "me muevo bien",
        "sin problema",
        "sin problemas",
        "estoy caminando",
        "mejor",
        "mejorando",
        "fuerte",
        "con fuerza",
        "sin ayuda",
        "yo solo",
        "yo sola",
        "independiente",
        "bien de fuerzas",
    ],
}

# ---------------------------------------------------------------------------
# Negation markers — when these appear immediately before a symptom keyword
# (within ~3 tokens), the keyword is **not** counted.
# ---------------------------------------------------------------------------

NEGATION_MARKERS: tuple[str, ...] = (
    "no",
    "sin",
    "nada de",
    "ningún",
    "ningun",
    "ninguna",
    "ninguno",
    "nunca",
    "jamás",
    "jamas",
    "cero",
    "ausencia de",
    "ausente",
    "libre de",
    "carezco de",
    "no tengo",
    "no he tenido",
    "no he sentido",
    "no presento",
    "no he presentado",
    "no hay",
    "no hubo",
)

# ---------------------------------------------------------------------------
# Numeric thresholds (cross-checked independently of keyword matching)
# ---------------------------------------------------------------------------

# Pain NRS score (0-10).  Values >= RED trigger RED; >= YELLOW trigger YELLOW.
PAIN_RED_THRESHOLD: int = 8
PAIN_YELLOW_THRESHOLD: int = 5

# Temperature in °C.  Values >= RED trigger RED; >= YELLOW trigger YELLOW.
TEMP_RED_THRESHOLD: float = 38.5
TEMP_YELLOW_THRESHOLD: float = 37.5

# ---------------------------------------------------------------------------
# Ambiguity / uncertainty phrases (patient unclear → one clarification question)
# ---------------------------------------------------------------------------

AMBIGUITY_PHRASES: tuple[str, ...] = (
    "no sé",
    "no se",
    "mas o menos",
    "más o menos",
    "regular",
    "ahi",
    "ahí",
    "pues",
    "no estoy seguro",
    "no estoy segura",
    "tal vez",
    "quizás",
    "quizas",
    "no recuerdo",
    "no me acuerdo",
    "puede ser",
    "a veces",
    "no sabría decir",
    "no sabria decir",
    "difícil de decir",
    "dificil de decir",
    "ni bien ni mal",
    "mitad y mitad",
)

# ---------------------------------------------------------------------------
# Cross-cutting red-flag keywords — these are checked **regardless of domain**
# and always trigger RED.
# ---------------------------------------------------------------------------

CROSS_CUTTING_RED_FLAGS: tuple[str, ...] = (
    # Cardiopulmonary
    "dolor en el pecho",
    "opresión en el pecho",
    "opresion en el pecho",
    "infarto",
    "ataque al corazón",
    "ataque al corazon",
    # Respiratory
    "no puedo respirar",
    "dificultad para respirar",
    "me ahogo",
    "falta de aire grave",
    # Neurological
    "no puedo hablar",
    "no siento la cara",
    "parálisis facial",
    "paralisis facial",
    "pérdida del conocimiento",
    "perdida del conocimiento",
    "me desmayé",
    "me desmayo",
    "quedé inconsciente",
    "quede inconsciente",
    "convulsión",
    "convulsion",
    "convulsiones",
    # Vascular
    "pierna hinchada y roja",
    "pantorrilla hinchada",
    "dolor en la pantorrilla",
    "no orino",
    "no he orinado",
    "no orina",
    # Hemorrhage
    "sangrado abundante",
    "hemorragia",
    "no para de sangrar",
    "sangre en la orina",
    "vómito con sangre",
    "vomito con sangre",
    "sangre en las heces",
    "heces negras",
    "heces con sangre",
    # Sepsis indicators
    "fiebre alta con confusión",
    "fiebre alta con confusión",
    "me siento muy mal",
    "siento que me muero",
    "creo que me estoy muriendo",
    "no me siento bien del todo",
    # Suicidal / self-harm
    "me quiero morir",
    "no quiero vivir",
    "me quiero hacer daño",
    "me quiero hacer dano",
    "ya no aguanto más",
    "ya no aguanto mas",
)

# ---------------------------------------------------------------------------
# Domain dispatch — maps a Spanish domain keyword to the canonical key.
# The rule engine lowercases the domain argument so this works case-insensitively.
# ---------------------------------------------------------------------------

DOMAIN_DISPATCH: dict[str, str] = {
    "dolor": DOMAIN_DOLOR,
    "fiebre": DOMAIN_FIEBRE,
    "herida": DOMAIN_HERIDA,
    "apetito": DOMAIN_APETITO,
    "sueño": DOMAIN_SUENO,
    "sueno": DOMAIN_SUENO,
    "movilidad": DOMAIN_MOVILIDAD,
}
