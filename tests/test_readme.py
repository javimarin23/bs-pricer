"""
Ejecuta literalmente los ejemplos de la sección Usage del README.

Si cambia una firma pública y el README no se actualiza (o al revés), la
suite falla. Los bloques se ejecutan en orden y en un mismo espacio de
nombres, porque el segundo reutiliza variables del primero, igual que haría
un lector copiándolos en una sesión.
"""

import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def bloques_python_de_seccion(texto, titulo):
    """Bloques ```python``` entre '## <titulo>' y el siguiente '## '."""
    seccion = re.search(rf"^## {re.escape(titulo)}\s*$(.*?)(?=^## |\Z)", texto, re.M | re.S)
    assert seccion is not None, f"README sin sección '## {titulo}'"
    return re.findall(r"^```python\s*\n(.*?)^```", seccion.group(1), re.M | re.S)


def test_readme_usage_se_ejecuta():
    bloques = bloques_python_de_seccion(README.read_text(encoding="utf-8"), "Usage")

    # Sin esta comprobación, renombrar la sección o cambiar la etiqueta del
    # bloque dejaría el test en verde sin ejecutar nada.
    assert len(bloques) >= 2

    espacio = {}
    for i, codigo in enumerate(bloques):
        exec(compile(codigo, f"README.md[Usage, bloque {i}]", "exec"), espacio)
