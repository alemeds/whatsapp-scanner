# 🔍 Analizador de Conversaciones WhatsApp

## ¿Qué hace esta aplicación?

Aplicación web (Streamlit) que analiza conversaciones exportadas de WhatsApp para detectar patrones específicos mediante diccionarios de términos clasificados por riesgo:

- **Acoso Sexual**
- **Cyberbullying**
- **Amenazas y Violencia**
- **Drogas**
- **Infidelidad**
- **Malas Palabras (Argentina)**
- **Robo y Estafas**
- **Suicidio y Autolesión**
- **Completo** (todas las categorías combinadas)

Estas categorías se pueden combinar entre sí, y además se les pueden sumar términos puntuales propios.

## 🚀 Instalación

### Requisitos
- Python 3.9+

### Pasos
```bash
git clone <url-del-repositorio>
cd whatsapp_analyzer_streamlit
python3 -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Ejecutar la aplicación
```bash
streamlit run whatsapp_analyzer_streamlit.py
```

La app abre automáticamente en `http://localhost:8501`.

## 📱 Cómo exportar chats de WhatsApp

### En Android
1. Abrí WhatsApp
2. Entrá al chat que querés analizar
3. Tocá los 3 puntos (⋮) → **Más** → **Exportar chat**
4. Seleccioná **"Sin archivos multimedia"** ⚠️ **Importante**
5. Guardá el archivo `.txt`

### En iPhone
1. Abrí WhatsApp
2. Entrá al chat que querés analizar
3. Tocá el nombre del contacto/grupo
4. Deslizá hacia abajo → **Exportar chat**
5. Seleccioná **"Sin archivos multimedia"** ⚠️ **Importante**
6. Guardá el archivo `.txt`

## 🎯 Cómo usar la aplicación

### 1. Elegir categorías de detección
En la barra lateral, **"🎯 Tipo(s) de Detección"** es un selector múltiple: podés elegir una o varias categorías predefinidas a la vez (por ejemplo, "Drogas" + "Amenazas y Violencia"). Sus términos se combinan sin duplicarse.

### 2. (Opcional) Agregar términos puntuales
En **"📁 Agregar Términos Puntuales"** podés subir un `.csv` o `.txt` propio con el formato `término,categoría`. Esos términos se suman a las categorías elegidas en el paso anterior (no las reemplazan). Si no elegís ninguna categoría predefinida, este archivo funciona como diccionario completo.

### 3. Configurar sensibilidad
- **Baja** (umbral 0.75): menos falsos positivos, detecta solo casos muy claros
- **Media** (umbral 0.60): balance entre precisión y detección (recomendado)
- **Alta** (umbral 0.45): más sensible, puede generar más falsos positivos

También se puede definir un **umbral personalizado** (0.0 a 1.0) que reemplaza el de la sensibilidad elegida.

### 4. Subir el archivo de chat
Subí el `.txt` exportado de WhatsApp. La app extrae automáticamente fecha, remitente y mensaje de cada línea.

### 5. Revisar resultados
- **Estadísticas**: total de mensajes, detecciones, porcentaje, riesgo promedio
- **Visualizaciones**: distribución de riesgo y detecciones por remitente
- **Evidencias**: lista filtrable (por remitente y riesgo mínimo) de los mensajes detectados, con los términos que matchearon

### 6. Descargar reportes
- **CSV completo**: todos los mensajes con su puntuación
- **CSV de detecciones**: solo los mensajes marcados como `DETECTADO`

## 📖 Formato de diccionario personalizado

Archivo `.csv` o `.txt`, una línea por término:
```
término,categoría
```

### Categorías reconocidas
| Categoría en el archivo | Se clasifica como |
|---|---|
| `palabras_alta` | Alto riesgo |
| `palabras_media` | Riesgo medio |
| `frases_contexto` | Frase de contexto |
| `contexto_laboral`, `contexto_relacion`, `contexto_financiero`, `contexto_agresion`, `contexto_emocional`, `contexto_digital`, `contexto_sustancias` | Contexto (cualquiera de estas) |

Las líneas vacías y las que empiezan con `#` se ignoran (sirven como comentarios).

### Ejemplo
```csv
# Diccionario para detectar bullying escolar
tonto,palabras_alta
idiota,palabras_alta
nadie te quiere,frases_contexto
colegio,contexto_digital
```

## ⚙️ Cómo se calcula el riesgo

Por cada mensaje se cuentan las coincidencias en cada categoría (alto riesgo, riesgo medio, frases de contexto, contexto). Esa cantidad se convierte en una proporción que **satura al llegar a 3 coincidencias** (no depende del tamaño del diccionario), se multiplica por el peso de la categoría según la sensibilidad elegida, y se suman bonificaciones cuando se combinan alto riesgo + contexto o alto riesgo + contexto laboral. El resultado final es un valor entre 0.0 y 1.0.

Un mensaje se marca `DETECTADO` cuando su puntuación supera el umbral configurado.

## 🛡️ Consideraciones legales y éticas

- Obtené consentimiento antes de analizar conversaciones privadas de terceros.
- Respetá las leyes de privacidad locales.
- Usalo solo para fines legítimos (investigación, seguridad, protección).
- Todos los archivos se procesan localmente, en memoria — la app no persiste datos en disco ni los envía a servidores externos.
- Los resultados son indicativos, no pruebas definitivas: pueden ocurrir falsos positivos y falsos negativos. Verificá siempre las evidencias manualmente.

## 🔧 Solución de problemas

**El archivo no se carga / no se extraen mensajes:**
- Verificá que sea un `.txt` exportado de WhatsApp con la opción "Sin archivos multimedia".
- El formato esperado por línea es similar a `12/06/24, 10:30 a. m. - Nombre: mensaje` (o variantes con corchetes/AM-PM).
- Por diseño, se requieren más de 5 mensajes para que la app reconozca el patrón de exportación.

**Muchos falsos positivos:**
- Bajá la sensibilidad a "Baja" o subí el umbral personalizado.

**Pocos resultados:**
- Subí la sensibilidad a "Alta" o bajá el umbral personalizado.
- Revisá que las categorías elegidas (o tu diccionario puntual) incluyan los términos relevantes.

## 🆘 Recursos en casos de riesgo suicida

- 🚨 **Argentina**: 135 (Centro de Asistencia al Suicida)
- 🚨 **México**: 800-290-0024 (SAPTEL)
- 🚨 **España**: 717-003-717 (Teléfono de la Esperanza)
- 🚨 **Chile**: 4141 (Salud Responde)

---

*Este software se proporciona con fines educativos y de investigación. El usuario es responsable del uso ético y legal de la herramienta.*
