# Logs y Trazabilidad

Sistema de trazabilidad híbrida con triple salida: consola, fichero y base de datos.

::: src.gestor_logs.GestorLogs
    options:
      show_source: false
      members:
        - registrar

---

## Ejemplo de uso

```python
from src.gestor_logs import GestorLogs
from src.gestor_base_datos import GestorBaseDatos

db = GestorBaseDatos()
log = GestorLogs(gestor_db=db)

log.registrar("MAIN", "Pipeline iniciado")
log.registrar("ML", "Modelo entrenado", estado="EXITO")
log.registrar("ML", "Error en datos", estado="ERROR")
```

---

## Destinos de salida

| Destino | Descripción |
|---------|-------------|
| Consola | Salida en tiempo real con formato `[HH:MM:SS] [MODULO] mensaje` |
| Fichero | Archivo diario en `logs/ejecucion_YYYYMMDD.log` con formato estándar |
| SQLite | Tabla `logs_ejecucion` para auditoría estructurada |
