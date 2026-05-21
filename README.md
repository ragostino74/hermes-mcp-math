# Hermes MCP Math Server

Hermes MCP Server per il calcolo scientifico — v2.1.0

Strumenti puri di matematica (SymPy, NumPy, SciPy) esposti come MCP tools.
Nessun web search, nessun REST bridge — solo 9 tool essenziali.

## Tool disponibili

| Tool | Descrizione |
|------|-------------|
| `solve_equation` | Risolve equazioni algebriche (lineari, quadratiche, sistemi) |
| `differentiate` | Derivate prime, seconde e parziali |
| `integrate` | Integrali definiti e non definiti |
| `limit_func` | Calcolo di limiti di funzioni |
| `simplify_expr` | Semplificazione di espressioni simboliche |
| `symbolic_calculate` | Calcoli simbolici con valutazione numerica |
| `numerical_calculate` | Calcoli numerici complessi (NumPy, AST-safe eval) |
| `matrix_operations` | Operazioni matriciali (det, autovalori, inversa, SVD) |
| `statistics` | Statistica descrittiva e regressioni (NumPy/SciPy) |

## Caratteristiche

- Trasporto stdio (Claude Desktop, VS Code, Hermes Agent)
- Trasporto HTTP/StreamableHTTP (opzionale via `HERMES_MCP_TRANSPORT=http`)
- DNS rebinding protection abilitata
- Input validation anti-iniezione per tool SymPy
- Valutazione numerica sicura con AST parser + whitelist NumPy
- CORS configurabile via `HERMES_MCP_CORS_ORIGINS`
- Dipendenze minime: SymPy, NumPy, SciPy

## Installazione

```bash
pip install -r requirements.txt
```

## Esecuzione

```bash
# STDIO (default)
python hermes_mcp_math.py

# HTTP/StreamableHTTP
HERMES_MCP_TRANSPORT=http HERMES_MCP_PORT=18762 python hermes_mcp_math.py

# HTTP con CORS personalizzato
HERMES_MCP_TRANSPORT=http HERMES_MCP_PORT=18762 HERMES_MCP_CORS_ORIGINS="http://localhost,https://myapp.example.com" python hermes_mcp_math.py
```
