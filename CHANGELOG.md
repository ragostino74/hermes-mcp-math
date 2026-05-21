# Changelog

Tutti i cambiamenti notevoli a questo progetto saranno documentati in questo file.

Il formato è basato su [Keep a Changelog](https://keepachangelog.com/it/1.1.0/),
e il versionamento segue [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
