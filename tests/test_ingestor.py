from src.config import config
from src.ingestor_datos import IngestorDatos


class TestIngestorDatos:
    """Tests de ingesta, centrados en la detección de codificación (Espiral 3)."""

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

        ingestor = IngestorDatos(ruta_data=str(tmp_path))
        datasets = ingestor.leer_datos()

        assert "notas_bimestre" in datasets
        assert list(datasets["notas_bimestre"].columns) == ["n_siu", "Estudio", "nota_m"]

    def test_tiene_bom_utf8(self, tmp_path):
        con_bom = tmp_path / "con_bom.csv"
        con_bom.write_bytes(b"\xef\xbb\xbfn_siu;estudio\n")
        sin_bom = tmp_path / "sin_bom.csv"
        sin_bom.write_bytes(b"n_siu;estudio\n")

        assert IngestorDatos._tiene_bom_utf8(str(con_bom)) is True
        assert IngestorDatos._tiene_bom_utf8(str(sin_bom)) is False
