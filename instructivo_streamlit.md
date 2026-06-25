# 🔍 Analizador de Conversaciones WhatsApp - Instructivo Completo

## ¿Qué hace esta aplicación?

Esta aplicación web analiza conversaciones exportadas de WhatsApp para detectar patrones específicos como:
- **Acoso Sexual** - Detecta lenguaje inapropiado y comportamiento de acoso
- **Cyberbullying** - Identifica insultos, humillaciones y acoso digital
- **Amenazas y Violencia** - Encuentra amenazas directas y lenguaje violento
- **Tráfico de Drogas** - Detecta jerga y patrones de narcotráfico
- **Estafas y Fraudes** - Identifica esquemas de estafa y actividades fraudulentas
- **Ideación Suicida** - Detecta señales de riesgo suicida (para intervención)
- **Infidelidad** - Encuentra evidencias de comportamiento infiel

## 🚀 Instalación y Configuración

### Paso 1: Instalar Python
- Descargar Python 3.8+ desde [python.org](https://python.org)
- Durante la instalación, marcar "Add Python to PATH"

### Paso 2: Instalar dependencias
```bash
pip install streamlit pandas plotly
```

### Paso 3: Ejecutar la aplicación
```bash
streamlit run analizador_whatsapp.py
```

## 📱 Cómo exportar chats de WhatsApp

### En Android:
1. Abre WhatsApp
2. Ve al chat que quieres analizar
3. Toca los 3 puntos (⋮) → **Más** → **Exportar chat**
4. Selecciona **"Sin archivos multimedia"** ⚠️ **IMPORTANTE**
5. Guarda el archivo `.txt`

### En iPhone:
1. Abre WhatsApp
2. Ve al chat que quieres analizar
3. Toca el nombre del contacto/grupo
4. Desliza hacia abajo → **Exportar chat**
5. Selecciona **"Sin archivos multimedia"** ⚠️ **IMPORTANTE**
6. Guarda el archivo `.txt`

## 🎯 Cómo usar la aplicación

### 1. Seleccionar tipo de detección
En la barra lateral, elige:
- **Acoso Sexual, Cyberbullying, etc.** - Diccionarios predefinidos
- **Diccionario Personalizado** - Sube tu propio diccionario

### 2. Configurar sensibilidad
- **Baja**: Menos falsos positivos, detecta solo casos muy claros
- **Media**: Balance entre precisión y detección (recomendado)
- **Alta**: Más sensible, puede generar más falsos positivos

### 3. Subir archivo de chat
- Arrastra o selecciona el archivo `.txt` exportado de WhatsApp
- La aplicación extraerá automáticamente los mensajes

### 4. Revisar resultados
- **Estadísticas**: Total de mensajes, detecciones, porcentajes
- **Visualizaciones**: Gráficos de distribución y por remitente
- **Evidencias**: Lista detallada de mensajes detectados

### 5. Descargar reportes
- **CSV Completo**: Todos los mensajes con sus puntuaciones
- **Solo Detecciones**: Únicamente los mensajes problemáticos

## 📖 Crear diccionarios personalizados

### Formato de archivo
Crea un archivo `.csv` o `.txt` con el formato:
```
término,categoría
```

### Categorías disponibles:
- `palabras_alta` - Términos de alto riesgo
- `palabras_media` - Términos de riesgo medio
- `frases_contexto` - Frases que indican contexto sospechoso
- `contexto_*` - Contexto específico (laboral, emocional, digital, etc.)

### Ejemplo de diccionario personalizado:
```csv
# Diccionario para detectar bullying escolar
tonto,palabras_alta
idiota,palabras_alta
estúpido,palabras_alta
nadie te quiere,frases_contexto
eres un perdedor,frases_contexto
patio,contexto_laboral
recreo,contexto_laboral
colegio,contexto_laboral
```

## ⚙️ Configuración avanzada

### Ajustar umbral de detección
- **0.0-0.3**: Muy sensible (muchos falsos positivos)
- **0.4-0.6**: Equilibrado (recomendado)
- **0.7-1.0**: Muy estricto (solo casos muy claros)

### Filtrar resultados
- Por remitente específico
- Por nivel de riesgo mínimo
- Por rango de fechas (si disponible)

## 🛡️ Consideraciones legales y éticas

### ⚖️ Uso responsable:
- Obtén consentimiento antes de analizar conversaciones privadas
- Respeta las leyes de privacidad locales
- Usa solo para fines legítimos (investigación, seguridad, protección)

### 🔒 Privacidad y seguridad:
- Todos los archivos se procesan localmente
- No se almacenan datos en servidores externos
- Los archivos se eliminan automáticamente al cerrar la sesión
- Mantén confidenciales los resultados del análisis

### ⚠️ Limitaciones importantes:
- Los resultados son indicativos, no pruebas definitivas
- Pueden ocurrir falsos positivos (detecciones incorrectas)
- Pueden ocurrir falsos negativos (casos no detectados)
- Siempre verifica manualmente las evidencias

## 🔧 Solución de problemas

### El archivo no se carga:
- ✅ Verifica que sea un `.txt` exportado de WhatsApp
- ✅ Asegúrate de haber seleccionado "Sin archivos multimedia"
- ✅ El archivo debe contener fechas y nombres de contactos
- ✅ Prueba con diferentes codificaciones si hay caracteres especiales

### No se detectan mensajes:
- ✅ Revisa el formato de exportación de WhatsApp
- ✅ Verifica que los mensajes tengan el formato: `[fecha] - Nombre: mensaje`
- ✅ Comprueba que no esté vacío o corrupto

### Muchos falsos positivos:
- 🔧 Reduce la sensibilidad a "Baja"
- 🔧 Aumenta el umbral personalizado (0.7-0.8)
- 🔧 Revisa y ajusta tu diccionario personalizado

### Pocos resultados:
- 🔧 Aumenta la sensibilidad a "Alta"
- 🔧 Reduce el umbral personalizado (0.3-0.5)
- 🔧 Verifica que tu diccionario incluye términos relevantes

## 📊 Interpretación de resultados

### Puntuación de riesgo:
- **0.8-1.0**: Muy alto riesgo - Requiere atención inmediata
- **0.6-0.79**: Alto riesgo - Revisar cuidadosamente
- **0.4-0.59**: Riesgo medio - Evaluar contexto
- **0.2-0.39**: Riesgo bajo - Posible falso positivo
- **0.0-0.19**: Sin riesgo detectado

### Indicadores de calidad:
- **Alto % de detección + Baja puntuación promedio**: Posibles falsos positivos
- **Bajo % de detección + Alta puntuación promedio**: Detecciones muy precisas
- **Distribución uniforme**: Diccionario bien calibrado

## 🎛️ Personalización avanzada

### Crear diccionarios específicos:

#### Para ambiente laboral:
```csv
acoso,palabras_alta
hostigamiento,palabras_alta
promoción a cambio,frases_contexto
después del trabajo,frases_contexto
oficina,contexto_laboral
jefe,contexto_laboral
```

#### Para cyberbullying escolar:
```csv
nerd,palabras_media
fracasado,palabras_alta
todos se burlan,frases_contexto
en el recreo,contexto_digital
colegio,contexto_digital
```

#### Para detección de drogas:
```csv
hierba,palabras_alta
maría,palabras_alta
porro,palabras_alta
tienes algo,frases_contexto
parque,contexto_sustancias
dealer,contexto_sustancias
```

## 📈 Casos de uso recomendados

### 🏢 Empresas:
- Investigación de denuncias de acoso laboral
- Monitoreo de comunicaciones corporativas
- Prevención de harassment en equipos

### 🏫 Instituciones educativas:
- Detección de cyberbullying entre estudiantes
- Identificación de situaciones de riesgo
- Prevención de suicidio adolescente

### 👨‍⚖️ Ámbito legal:
- Recopilación de evidencias para casos judiciales
- Análisis forense de comunicaciones
- Investigaciones de fraude

### 👨‍👩‍👧‍👦 Uso familiar:
- Protección de menores online
- Detección de contenido inapropiado
- Supervisión parental responsable

## 🆘 Recursos adicionales

### Si detectas riesgo suicida:
- 🚨 **Argentina**: 135 (Centro de Asistencia al Suicida)
- 🚨 **México**: 800-290-0024 (SAPTEL)
- 🚨 **España**: 717-003-717 (Teléfono de la Esperanza)
- 🚨 **Chile**: 4141 (Salud Responde)

### Para casos de cyberbullying:
- Documenta toda la evidencia
- Contacta a las autoridades escolares
- Reporta en las plataformas sociales
- Busca apoyo psicológico

### Para casos legales:
- Consulta con abogado especializado
- Preserve la cadena de custodia
- No modifiques los archivos originales
- Documenta fecha y hora de análisis

## 🔄 Actualizaciones y mantenimiento

### Actualizar la aplicación:
```bash
git pull origin main
pip install --upgrade -r requirements.txt
```

### Actualizar diccionarios:
- Los diccionarios pueden actualizarse sin reinstalar
- Guarda copias de tus diccionarios personalizados
- Revisa periódicamente nuevos términos y jerga

### Reportar problemas:
- Documenta el error exacto
- Incluye el archivo de ejemplo (sin datos sensibles)
- Especifica tu sistema operativo y versión de Python

---

## 📞 Soporte técnico

**Email**: soporte@analizador-whatsapp.com  
**Documentación**: [docs.analizador-whatsapp.com](http://docs.analizador-whatsapp.com)  
**GitHub**: [github.com/analizador-whatsapp](http://github.com/analizador-whatsapp)

---

*Este software se proporciona con fines educativos y de investigación. El usuario es responsable del uso ético y legal de la herramienta.*