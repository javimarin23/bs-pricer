# Repo de pricing de opciones — instrucciones para Claude Code

## Contexto
Proyecto de portfolio para Santander CIB (Global Markets, Quants). Python.
Contenido: Black–Scholes analítico con griegas, Monte Carlo bajo GBM,
reducción de varianza y estudio de convergencia.
En la entrevista me van a preguntar por cada línea del núcleo numérico, así que
ese código lo escribo yo.

## Reparto de trabajo (regla principal)
- NO escribas el núcleo numérico salvo que te diga literalmente "escríbelo tú":
  fórmulas de pricing, griegas, estimadores Monte Carlo, variables antitéticas
  y de control, esquemas de discretización.
- En el núcleo haces code review duro: errores numéricos, sesgo del estimador,
  estabilidad, vectorización, casos límite (T→0, σ→0, deep ITM/OTM).
  Señala el problema y dame una pista. No me des la solución completa salvo
  que la pida.
- Sí puedes escribir tú: tests, fixtures, scripts de figuras, CLI, CI,
  empaquetado, docstrings, README y refactors de fontanería.

## Validación obligatoria antes de dar algo por bueno
- `pytest` en verde.
- Paridad put–call dentro de la tolerancia numérica.
- MC frente a analítico: el error ha de caer dentro de ~3 errores estándar y
  escalar como 1/√N.
- Griegas por diferencias finitas frente a las analíticas.
- Semillas fijadas en todo lo estocástico.

## Estilo
- Type hints, NumPy vectorizado y sin bucles Python en caminos calientes.
- Commits pequeños con mensaje en inglés, en imperativo.
- Nada de dependencias nuevas sin preguntarme.
