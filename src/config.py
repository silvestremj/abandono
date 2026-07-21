import os
from typing import Any, Dict

import yaml


class ConfigLoader:
    """Clase para cargar y distribuir la configuración del proyecto."""

    def __init__(self, config_file: str = "config.yaml") -> None:
        # Buscamos el yaml en la raíz del proyecto (un nivel por encima de src)
        base_dir: str = os.path.dirname(os.path.dirname(__file__))
        self.config_path: str = os.path.join(base_dir, config_file)
        self.config_data: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Lee el archivo YAML de forma segura."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            print(
                f" Error: No se encontró el archivo de configuración en {self.config_path}"
            )
            return {}
        except yaml.YAMLError as exc:
            print(f" Error al leer el archivo YAML: {exc}")
            return {}

    @property
    def datasets(self) -> Dict[str, Any]:
        return self.config_data.get("datasets", {})

    @property
    def paths(self) -> Dict[str, Any]:
        return self.config_data.get("paths", {})

    @property
    def reglas_riesgo(self) -> Dict[str, Any]:
        return self.config_data.get("reglas_riesgo", {})

    @property
    def machine_learning(self) -> Dict[str, Any]:
        """Devuelve el diccionario con la configuración de Machine Learning."""
        return self.config_data.get("machine_learning", {})


# Instancia global para que los demás módulos la importen directamente
config = ConfigLoader()
