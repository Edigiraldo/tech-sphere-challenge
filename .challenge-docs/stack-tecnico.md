# Stack técnico — propuesta de stack abierto

Este documento reúne el conjunto de herramientas propuesto para el reto: piezas abiertas
o con nivel gratuito que eliminan la barrera del costo, para que la competencia se decida
por la **arquitectura y la experiencia de usuario** y no por el presupuesto de cada
participante.

**El stack es abierto con una sola excepción: el modelo de lenguaje** (§1). Orquestación,
voz, RAG y embeddings los eliges tú; las herramientas que siguen son sugerencias, no
obligaciones, y puedes usar otras si lo consideras.

---

## 1. Los modelos permitidos

El modelo de lenguaje que razona en tu agente debe ser **uno de estos**:

| Modelo | Dónde corre | Detalle |
|---|---|---|
| **Google Gemini 1.5 Flash** | Nube, nivel gratuito | [§2](#2-inferencia-en-la-nube-niveles-gratuitos) |
| **Llama 3.1 70B** (vía Groq) | Nube, nivel gratuito | [§2](#2-inferencia-en-la-nube-niveles-gratuitos) |
| **Llama 3.2** (1B o 3B) | Local, CPU | [§3](#3-modelos-locales-para-cpu) |
| **Phi-3.5 Mini** (3.8B) | Local, CPU | [§3](#3-modelos-locales-para-cpu) |

Elige el que prefieras según tu arquitectura. **Tu informe final debe declarar cuál
usaste y por qué lo elegiste.** Usar un modelo fuera de esta lista descalifica la entrega
(compuerta G3 de la [rúbrica](rubrica-evaluacion.md#g3--usas-uno-de-los-modelos-permitidos)).

La lista es cerrada porque el costo del modelo no debe decidir el reto: con las mismas
opciones sobre la mesa, la diferencia la hace la ingeniería.

Lo demás no está restringido. El reconocimiento de voz, la síntesis de voz, la base
vectorial, los embeddings y el framework de orquestación son decisión tuya, uses o no las
herramientas de este documento.

---

## 2. Inferencia en la nube (niveles gratuitos)

Para razonamiento complejo o ventanas de contexto grandes sin hardware local costoso.

### Google Gemini 1.5 Flash — 15 RPM gratis

Su ventaja competitiva es la **ventana de contexto de 1 millón de tokens**: permite
cargar múltiples guías de práctica clínica, protocolos de triaje y el historial completo
del paciente en una sola consulta, sin fragmentar la información en exceso, lo que
preserva la coherencia del razonamiento médico.

El nivel gratuito de Google AI Studio ofrece **15 solicitudes por minuto**, suficiente
para desarrollar y para la demostración en vivo.

→ [Google AI Studio](https://aistudio.google.com/)

### Groq Cloud — latencia ultra-baja

Fundamental cuando la prioridad es la fluidez de la conversación. Sus unidades de
procesamiento de lenguaje (LPU) entregan tokens a velocidad casi instantánea y eliminan
el lag de la interacción.

Da acceso gratuito a modelos potentes como **Llama 3.1 70B** y, sobre todo, a **Whisper
Large V3** para transcripción de voz a texto. Procesar el audio en milisegundos permite
que el agente responda casi en cuanto el paciente termina de hablar.

→ [Consola de Groq (Llama & Whisper)](https://console.groq.com/)

---

## 3. Modelos locales para CPU

Modelos de lenguaje pequeños (SLM) optimizados para correr en computadores comunes, sin
GPU dedicada.

### Llama 3.2 (1B y 3B)

Los modelos más eficientes de Meta para computación de borde. El de **1B parámetros
consume ~1.2 GB de RAM**, lo que permite resumir notas clínicas y hacer triaje básico de
forma 100 % privada y local, incluso en laptops de gama media-baja.

→ [Descargar vía Ollama](https://ollama.com/library/llama3.2)

### Phi-3.5 Mini (3.8B)

El modelo de Microsoft diseñado para razonamiento lógico superior. Pese a su tamaño,
compite con modelos dos o tres veces más grandes en capacidad de seguir instrucciones
complejas y de adherirse a protocolos médicos estrictos sin desviarse.

→ [Ver en Hugging Face](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)

### Ollama — orquestador

La pieza que vuelve trivial correr modelos locales: gestiona la descarga y la
cuantización, y expone una API local compatible con el estándar de OpenAI, lo que
facilita integrarla con cualquier interfaz web o móvil.

→ [Instalar Ollama](https://ollama.com/)

---

## 4. Gestión de conocimiento médico (RAG)

El modelo no necesita entrenamiento médico: necesita acceso a fuentes confiables. El RAG
le permite "leer" guías oficiales en tiempo real.

### ChromaDB — local y gratis

Base de datos vectorial de código abierto que corre localmente. Permite indexar miles de
páginas de literatura médica, vademécums y protocolos de emergencia sin costo de
servidores. Es ligera y se integra con Python o JavaScript.

→ [Documentación de ChromaDB](https://www.trychroma.com/)

### BGE-M3 — embeddings en español

El componente crítico para la precisión. BGE-M3 es un modelo de embeddings multilingüe
que sobresale en español: entiende sinónimos médicos y conceptos complejos en nuestro
idioma, lo que asegura que lo recuperado del RAG sea realmente relevante para la consulta
del paciente.

→ [Ver BGE-M3 en Hugging Face](https://huggingface.co/BAAI/bge-m3)

### El flujo de conocimiento

1. Consulta del paciente en español.
2. BGE-M3 busca en ChromaDB el protocolo pertinente.
3. Se inyecta el texto médico recuperado al modelo.
4. Respuesta fundamentada, sin alucinaciones.

---

## 5. Interfaces de voz en español

Alternativas locales y gratuitas a servicios comerciales, optimizadas para la prosodia y
la acentuación del español médico.

### Kokoro-82M — alta calidad

Una revelación en síntesis de voz (TTS): pese a su tamaño mínimo, ofrece una calidad que
rivaliza con modelos comerciales pesados. Soporta voces en español nativo que manejan
correctamente la entonación clínica y, por lo ligero, genera audio en tiempo real sin GPU
potente. Útil para que el agente suene empático y profesional al dar instrucciones.

→ [Demo en español](https://huggingface.co/spaces/leonelhs/kokoro-tts-spanish) ·
[Repositorio base](https://huggingface.co/hexgrad/Kokoro-82M)

### Piper — voces regionales, local-first

Diseñado para ser ultra-rápido en hardware limitado (desde una Raspberry Pi hasta una
laptop de oficina). Ofrece modelos preentrenados para acentos específicos de México y
España. Su ventaja principal es la **latencia mínima**: el audio empieza a reproducirse
casi en el mismo instante en que se genera el texto, algo vital para una conversación
fluida.

→ [Piper en GitHub](https://github.com/rhasspy/piper)

---

## 6. Viabilidad en hardware común

Los modelos recomendados (1B a 3B parámetros) corren en una laptop estándar de **8 a
16 GB de RAM**. No hace falta hardware de servidor especializado.

| Componente | RAM aproximada |
|---|---:|
| Sistema operativo | 3.2 GB |
| Llama 3.2 (1B) *o* Phi-3.5 Mini (3.8B) | 1.2 GB / 2.8 GB |
| Voz (Kokoro / Piper) | 0.6 GB |
| RAG (ChromaDB + aplicación) | 0.9 GB |

Los dos modelos locales son alternativas entre sí, no componentes simultáneos: corres uno
u otro.

**RAM mínima 8 GB · procesamiento en CPU · costo de APIs y modelos: $0 · arquitectura
abierta.**

---

Este stack es una base de referencia, no una imposición: fuera de la lista de modelos
permitidos, resuelve la arquitectura como prefieras. Lo que se evalúa es cómo resuelves
los retos de arquitectura, la precisión en la recuperación de información médica y la
calidad de la interacción con el paciente.
