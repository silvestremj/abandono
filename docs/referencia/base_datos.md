# Gestor de Base de Datos

Gestor de persistencia en SQLite. Almacena datos RAW, la tabla maestra y logs de ejecución.

::: src.gestor_base_datos.GestorBaseDatos
    options:
      show_source: false
      members:
        - inicializar_tablas_fijas
        - guardar_datos_raw
        - guardar_datos_master
        - registrar_log

---

## Ejemplo de uso

```python
from src.gestor_base_datos import GestorBaseDatos

db = GestorBaseDatos()
db.inicializar_tablas_fijas()

# Guardar datos
db.guardar_datos_raw(datasets)
db.guardar_datos_master(df_master)

# Registrar log
db.registrar_log("MODULO", "EXITO", "Operación completada")
```

---

## Tablas de SQLite

| Tabla | Contenido |
|-------|-----------|
| `logs_ejecucion` | Registro de eventos del pipeline (id, timestamp, modulo, estado, mensaje) |
| `raw_inscripciones` | Datos crudos de inscripciones |
| `raw_notas_bimestre` | Datos crudos de notas por bimestre |
| `raw_actual` | Datos crudos del archivo Excel actual |
| `master_estudiantes` | Tabla maestra consolidada |
