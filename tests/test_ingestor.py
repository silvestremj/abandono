import pandas as pd
import pytest

from src.config import config
from src.ingestor_datos import IngestorDatos


def crear_datasets_minimos(tmp_path, excepto=None):
    """Crea en ``tmp_path`` un fichero mínimo válido por cada dataset declarado.

    ``leer_datos`` exige que estén los tres ficheros de ``config.datasets``, así
    que las pruebas centradas en UNO de ellos necesitan que los otros dos
    existan. Con ``excepto`` se omite deliberadamente uno para provocar el
    error de fichero ausente.

    Args:
        tmp_path: Directorio temporal de la prueba.
        excepto: Clave de ``config.datasets`` que NO se debe crear.
    """
    contenido_csv = {
        "inscripciones": "n_siu;estudio\n1001;CESE\n",
        "notas_bimestre": (
            "n_siu;Estudio;nota_m;asist_m;Comentario\n"
            "1001;CESE;9,5;0,93;Bimestre: 1.0\n"
        ),
    }

    for clave, conf in config.datasets.items():
        if clave == excepto:
            continue
        ruta = tmp_path / conf["nombre"]
        if ruta.exists():
            continue
        if conf["tipo"] == "excel":
            pd.DataFrame({"n_siu": ["1001"], "estudio": ["CESE"]}).to_excel(
                ruta, index=False
            )
        else:
            ruta.write_bytes(
                contenido_csv[clave].encode(conf.get("encoding", "utf-8"))
            )


class TestIngestorDatos:
    """Tests de ingesta, centrados en la detección de codificación."""

    def test_lee_csv_con_bom_utf8_pese_a_encoding_latin1_declarado(self, tmp_path):
        """Un CSV con BOM UTF-8 pero configurado con encoding='latin1' en
        config.yaml no debe corromper la cabecera (p.ej. 'n_siu' ->
        'ï»¿n_siu'), lo que rompería cualquier normalización/cruce posterior
        sobre esa columna. Detectado empíricamente en
        LSE_Inscrip_Baja_Recibido.csv."""
        conf_inscripciones = config.datasets["inscripciones"]
        assert conf_inscripciones["encoding"] == "latin1"

        ruta = tmp_path / conf_inscripciones["nombre"]
        contenido = "n_siu;estudio\n1001;CESE\n"
        ruta.write_bytes(b"\xef\xbb\xbf" + contenido.encode("utf-8"))
        crear_datasets_minimos(tmp_path)

        ingestor = IngestorDatos(ruta_data=str(tmp_path))
        datasets = ingestor.leer_datos()

        assert "inscripciones" in datasets
        assert list(datasets["inscripciones"].columns) == ["n_siu", "estudio"]
        assert datasets["inscripciones"].loc[0, "n_siu"] == 1001

    def test_lee_csv_sin_bom_normalmente_con_encoding_declarado(self, tmp_path):
        """Un CSV latin1 real (con tildes/ñ), sin BOM, se sigue leyendo tal
        cual con el encoding declarado — la detección de BOM no debe alterar
        el caso normal."""
        conf_notas = config.datasets["notas_bimestre"]
        assert conf_notas["encoding"] == "latin1"

        ruta = tmp_path / conf_notas["nombre"]
        contenido = "n_siu;Estudio;nota_m\n1001;CEIoT;8,5\n"
        ruta.write_bytes(contenido.encode("latin1"))
        crear_datasets_minimos(tmp_path)

        ingestor = IngestorDatos(ruta_data=str(tmp_path))
        datasets = ingestor.leer_datos()

        assert "notas_bimestre" in datasets
        assert list(datasets["notas_bimestre"].columns) == ["n_siu", "Estudio", "nota_m"]

    def test_error_claro_si_falta_un_fichero_declarado(self, tmp_path):
        """Si falta uno de los ficheros de config.yaml, la ingesta debe parar
        con un error que nombre el fichero y la carpeta donde se esperaba.

        Antes devolvía el diccionario incompleto y el fallo estallaba mucho
        más tarde, en la preparación, como un ``KeyError: 'actual'`` que no le
        dice nada a quien solo quiere usar la herramienta. Como ``data/`` no
        se versiona, es lo que ve cualquiera que clone el repositorio."""
        crear_datasets_minimos(tmp_path, excepto="actual")
        nombre_ausente = config.datasets["actual"]["nombre"]

        ingestor = IngestorDatos(ruta_data=str(tmp_path))

        with pytest.raises(FileNotFoundError) as excinfo:
            ingestor.leer_datos()

        mensaje = str(excinfo.value)
        assert nombre_ausente in mensaje
        assert str(tmp_path) in mensaje

    def test_no_falla_si_estan_todos_los_ficheros_declarados(self, tmp_path):
        """Con los tres ficheros presentes, la ingesta devuelve las tres claves."""
        crear_datasets_minimos(tmp_path)

        datasets = IngestorDatos(ruta_data=str(tmp_path)).leer_datos()

        assert set(datasets) == set(config.datasets)

    def test_tiene_bom_utf8(self, tmp_path):
        con_bom = tmp_path / "con_bom.csv"
        con_bom.write_bytes(b"\xef\xbb\xbfn_siu;estudio\n")
        sin_bom = tmp_path / "sin_bom.csv"
        sin_bom.write_bytes(b"n_siu;estudio\n")

        assert IngestorDatos._tiene_bom_utf8(str(con_bom)) is True
        assert IngestorDatos._tiene_bom_utf8(str(sin_bom)) is False
