# Changelog

Tutti i cambiamenti notevoli a questo progetto saranno documentati in questo file.

Il formato è basato su [Keep a Changelog](https://keepachangelog.com/it/1.1.0/),
e il versionamento segue [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.2.0] - 2026-05-24

### Corretti
- **Return type crash**: tutti i tool dichiarano ora `-> dict` invece di `-> str` (compatibile con FastMCP SDK >= 1.27)
- **Collisione nomi SymPy**: rinominati tutti gli import SymPy con prefisso `_sympy_` per prevenire conflitti con i nomi dei tool MCP (`integrate`, `simplify`, ecc.)
- **DNS rebinding protection**: passati correttamente i parametri di sicurezza a `FastMCP()`, host impostato su `0.0.0.0`
- **SciPy in requirements.txt**: aggiunta dipendenza mancante (usata da `statistics`)
- **AST attribute bypass**: aggiunto blocco per `ast.Attribute` nella whitelist NumPy-safe eval, prevenendo accesso a dunder/methods
- **System solver parsing**: migliorato il parsing di sistemi lineari con `_split_eq_string()` e `_extract_symbols_from_eqs()`, rispetto al fragile replace-based original
- **Type hint bounds**: corretto `lower_bound: float | None` invece di `float = None`

### Modificati
- Tutti i tool restituiscono direttamente `dict` — rimossi tutti i `json.dumps()` (FastMCP serializza in automatico)

## [2.1.0] - 2025-05-21

### Aggiunti
- Valutazione espressioni numeriche sicura con AST parser (sostituisce eval diretto)
- Validazione input per tool SymPy — blocco di pattern sospetti (doppio underscore, import, exec)
- CORS configurabile via variabile d'ambiente `HERMES_MCP_CORS_ORIGINS`

### Modificati
- `numerical_calculate`: da `eval()` ad AST parser + eval sicuro su whitelist NumPy
- Tool SymPy: validazione input con regex anti-iniezione
- Avvio HTTP: origins CORS da variabile d'ambiente (default: `http://localhost`)

## [2.0.0] - 2025-XX-XX

### Aggiunti
- 9 tool MCP matematici: solve_equation, differentiate, integrate, limit_func, simplify_expr, symbolic_calculate, numerical_calculate, matrix_operations, statistics
- Doppio trasporto: stdio (default) + HTTP/StreamableHTTP
- Supporto SymPy, NumPy, SciPy
