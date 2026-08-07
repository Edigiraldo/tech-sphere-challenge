# Ficha Técnica

Modelos sugeridos y stack abierto — no hay un LLM único obligatorio.
Opciones en la nube con tier gratuito (Gemini 1.5 Flash, Groq con Llama 3.1 70B) y modelos locales que corren en cualquier laptop (Llama 3.2, Phi-3.5 Mini vía Ollama), además de RAG (ChromaDB + BGE-M3) y voz (Kokoro, Piper).

## 01. APIs de Inferencia en la Nube (Tiers Gratuitos)

Para tareas que requieren un razonamiento complejo o una ventana de contexto masiva sin necesidad de hardware local costoso.

### Google Gemini 1.5 Flash - 15 RPM FREE

Esta herramienta es el 'cerebro' ideal para manejar la complejidad del reto. Su ventaja competitiva radica en su ventana de contexto de 1 millón de tokens. Esto permite cargar múltiples guías de práctica clínica, protocolos de triaje y el historial completo del paciente en una sola consulta. No hay necesidad de fragmentar la información (chunking) excesivamente, lo que preserva la coherencia del razonamiento médico. El nivel gratuito a través de Google AI Studio ofrece 15 solicitudes por minuto, más que suficiente para el desarrollo y la demostración en vivo del sistema.

### Groq Cloud - Latencia Ultra-Baja

Groq es fundamental cuando la prioridad es la fluidez de la conversación. Utiliza unidades de procesamiento de lenguaje (LPU) que entregan tokens a una velocidad casi instantánea, eliminando el 'lag' en la interacción. Proporciona acceso gratuito a modelos potentes como Llama 3.1 70B y, vitalmente, a Whisper Large V3 para la transcripción de voz a texto. Su capacidad de procesar audio en milisegundos permite que la IA responda al paciente casi al instante de que este termine de hablar, creando una experiencia humana y natural.

## 02. Modelos Locales (SLMs) para CPU
Modelos de lenguaje pequeño (Small Language Models) optimizados para ejecutarse en computadores comunes sin GPU dedicada.

### Llama 3.2 (1B & 3B)
Los modelos más eficientes de Meta para computación de borde. El modelo de 1B parámetros consume aproximadamente 1.2GB de RAM, lo que permite realizar resúmenes de notas clínicas y triaje básico de forma 100% privada y local, incluso en laptops de gama media-baja.

### Phi-3.5 Mini (3.8B)
El modelo de Microsoft diseñado para el razonamiento lógico superior. A pesar de su tamaño, compite con modelos 2 o 3 veces más grandes en la capacidad de seguir instrucciones complejas y adherirse a protocolos médicos estrictos sin desviarse.

### Ollama (Orquestador)
La pieza de software que hace que correr modelos locales sea trivial. Ollama gestiona la descarga, la cuantización y expone una API local compatible con el estándar de OpenAI, facilitando la integración con cualquier interfaz web o móvil.

## 03. Gestión de Conocimiento Médico (RAG)
El LLM no necesita entrenamiento médico; necesita acceso a fuentes confiables. El RAG permite que el modelo 'lea' guías oficiales en tiempo real.

### ChromaDB (Local & Gratis)
Una base de datos vectorial de código abierto que se ejecuta localmente. Permite indexar miles de páginas de literatura médica, vademécums y protocolos de emergencia sin costo de servidores. Es ligera y se integra perfectamente con Python o JavaScript.

### BGE-M3 (Embeddings en Español)
Este es el componente crítico para la precisión. BGE-M3 es un modelo de embeddings multilingüe que sobresale en español. Permite que el sistema entienda sinónimos médicos y conceptos complejos en nuestro idioma, asegurando que la información recuperada del RAG sea realmente relevante para la consulta del paciente.

## 04. Interfaces de Voz en Español (Sin Pay-to-Win)
Alternativas locales y gratuitas a servicios como ElevenLabs, optimizadas para la prosodia y acentuación del español médico.

### Kokoro-82M (Español) - Alta Calidad
Kokoro es una revelación en síntesis de voz (TTS). A pesar de su tamaño extremadamente pequeño, ofrece una calidad que rivaliza con modelos comerciales pesados. Soporta voces en español nativo que manejan correctamente la entonación clínica. Al ser tan ligero, puede generar audio en tiempo real sin necesidad de una GPU potente. Ideal para que el asistente de voz suene empático y profesional al dar instrucciones de tratamiento o triaje.

### Piper (Voces Regionales) - Local-First
Piper está diseñado para ser ultra-rápido y ejecutarse en dispositivos de hardware limitado (como una Raspberry Pi o una laptop de oficina). Ofrece modelos pre-entrenados para acentos específicos de México y España. Su principal ventaja es la latencia cero: el audio comienza a reproducirse casi en el mismo instante en que se genera el texto, vital para una conversación fluida.

## 05. Viabilidad en Hardware Común
Esta gráfica demuestra que los modelos recomendados (1B a 3B parámetros) pueden ejecutarse en una laptop estándar con 8GB-16GB de RAM. No se requiere hardware de servidor especializado; esto garantiza que el éxito del proyecto dependa de la creatividad y la implementación técnica del participante, no de su presupuesto.

### Consumo de RAM Estimado (GB)

| Componente | Consumo estimado (GB) |
| --- | ---: |
| Sistema Operativo | 3.2 |
| Llama 3.2 (1B) | 1.2 |
| Phi-3.5 Mini (3.8B) | 2.8 |
| Voz (Kokoro/Piper) | 0.6 |
| RAG (ChromaDB + App) | 0.9 |

> 8 GB - RAM Mínima
>
> CPU - Procesamiento
>
> $0 - Costo APIs / Modelos
>
> Open - Arquitectura
