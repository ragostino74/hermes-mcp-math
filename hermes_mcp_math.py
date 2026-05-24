#!/usr/bin/env python3
"""
Hermes MCP Math Server — Strumenti di calcolo matematico (SymPy, NumPy, SciPy)

Trasporto: stdio (default) + HTTP/StreamableHTTP (se HERMES_MCP_TRANSPORT=http)

Uso stdio:
  python3 hermes_mcp_math.py

Uso HTTP:
  HERMES_MCP_TRANSPORT=http HERMES_MCP_PORT=18762 python3 hermes_mcp_math.py
"""
import ast
import json, sys, os, re, asyncio, signal as sig_mod

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    print("ERROR: mcp package non installato. Installare con: pip install mcp", file=sys.stderr)
    sys.exit(1)

try:
    from sympy import (
        symbols, Eq, Matrix, solve as _sympy_solve,
        diff as _sympy_diff, integrate as _sympy_integrate,
        limit as sympy_limit, simplify as _sympy_simplify,
        factor as _sympy_factor, expand as _sympy_expand,
        sympify, I, pi, E, oo, sin, cos, tan, exp, log, sqrt, Abs,
        asin, acos, atan, sinh, cosh, tanh,
        factorial, binomial,
    )
    SYMPY_AVAILABLE = True
except ImportError as e:
    SYMPY_AVAILABLE = False
    print(f"WARNING: SymPy non disponibile ({e})", file=sys.stderr)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError as e:
    NUMPY_AVAILABLE = False
    print(f"WARNING: NumPy non disponibile ({e})", file=sys.stderr)
    np = None

try:
    from scipy import linalg as scipy_linalg
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError as e:
    SCIPY_AVAILABLE = False
    print(f"WARNING: SciPy non disponibile ({e})", file=sys.stderr)
    scipy_linalg = None
    scipy_stats = None

TRANSPORT = os.environ.get("HERMES_MCP_TRANSPORT", "stdio")

# CORS origins configurabili via variabile d'ambiente (default: localhost solo)
_CORS_ORIGINS_RAW = os.environ.get("HERMES_MCP_CORS_ORIGINS", "http://localhost")
CORS_ORIGINS = [o.strip() for o in _CORS_ORIGINS_RAW.split(",") if o.strip()]

mcp_server = FastMCP(
    name="hermes-math-mcp",
    host="0.0.0.0",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True),
)


# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES DI SICUREZZA
# ═══════════════════════════════════════════════════════════════════════════

_SYMPY_INJECTION_RE = re.compile(
    r"__|eval\(|exec\(|compile\(|open\(|getattr\(|setattr\(|delattr\("
    r"|__import__|globals\(\)|locals\(\)|hasattr\(|vars\(\)"
    r"|dir\(\)|breakpoint\(|print\(|input\(",
    re.IGNORECASE,
)


def _validate_sympy_input(value: str) -> str | None:
    """Valida input per tool SymPy. Restituisce None se OK, stringa errore altrimenti."""
    if _SYMPY_INJECTION_RE.search(value.strip()):
        return "Input potenzialmente pericoloso bloccato"
    return None


_NUMPY_WHITELIST = {
    # Modulo principale e alias
    "np", "numpy",
    # Costanti
    "pi", "e", "euler_gamma", "inf", "nan",
    # Trigonometriche + inverse
    "sin", "cos", "tan", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh",
    # Exp/log
    "exp", "log", "log10", "log2", "log1p", "expm1",
    # Radici/poteri
    "sqrt", "cbrt", "square", "power",
    # Arithmetiche
    "add", "subtract", "multiply", "divide", "true_divide",
    "floor_divide", "remainder", "mod", "divmod",
    # Arrotondamento
    "absolute", "fabs", "sign", "floor", "ceil", "trunc", "rint", "round",
    # Min/max
    "maximum", "minimum", "fmax", "fmin",
    # Array utility
    "array", "arange", "linspace", "logspace",
    "zeros", "ones", "empty", "full", "diag", "eye",
    # Statistiche
    "mean", "median", "std", "var", "sum", "prod",
    "min", "max", "ptp", "percentile",
    "nanmean", "nanstd", "nanvar", "nansum", "nanprod",
    # Complessi
    "real", "imag", "conj", "conjugate", "angle",
    # Comparazione
    "greater", "greater_equal", "less", "less_equal",
    "equal", "not_equal", "logical_and", "logical_or",
    "logical_not", "logical_xor",
}


def _safe_numpy_eval(expression: str) -> float | list | bool | complex:
    """Valuta espressione numerica con AST parser + eval su whitelist NumPy."""
    if _SYMPY_INJECTION_RE.search(expression):
        raise ValueError("Input potenzialmente pericoloso bloccato")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Espressione non valida (sintassi): {e}")

    for node in ast.walk(tree):
        # Blocca accesso ad attributi privati/__dunder__ per prevenire bypass
        if isinstance(node, ast.Attribute):
            raise ValueError("Accesso ad attributi non consentito")
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, complex, str, bool, type(None))):
                raise ValueError(f"Tipo costante non consentito: {type(node.value).__name__}")
        elif isinstance(node, ast.Name):
            if node.id not in _NUMPY_WHITELIST:
                raise ValueError(f"Nome non consentito: {node.id}")
        elif isinstance(node, ast.BinOp):
            allowed = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
            if not isinstance(node.op, allowed):
                raise ValueError(f"Operatore non consentito: {type(node.op).__name__}")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                raise ValueError(f"Operatore unario non consentito: {type(node.op).__name__}")
        elif isinstance(node, ast.Call):
            raise ValueError("Chiamate di funzione non consentite")
        elif isinstance(node, ast.Subscript):
            raise ValueError("Indicizzazione array non consentita")
        elif isinstance(node, (ast.List, ast.Tuple)):
            raise ValueError("Liste/tuple non consentite")

    local_vars = {"np": np, "numpy": np, "pi": np.pi, "e": np.e}
    result = eval(compile(tree, "<safe_numpy>", "eval"), {"__builtins__": {}}, local_vars)

    if isinstance(result, (np.floating, np.integer)):
        return float(result)
    elif isinstance(result, np.ndarray):
        return result.tolist()
    elif isinstance(result, (np.bool_, bool)):
        return bool(result)
    elif isinstance(result, np.complexfloating):
        return complex(result)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# CALCOLO SIMBOLICO (SymPy)
# ═══════════════════════════════════════════════════════════════════════════

@mcp_server.tool()
async def solve_equation(equation: str, variable: str = "x") -> dict:
    """Risolve equazioni algebriche (lineari, quadratiche, sistemi).

    Args:
        equation: Equazione. Esempi:
            - "x**2 - 4"          → trova le radici
            - "2*x + 3 = 7"       → risolve per x
            - "[x + y = 5, x - y = 1]" → sistema lineare
        variable: Variabile principale (default: "x"). Per sistemi, omessa.
    """
    if not SYMPY_AVAILABLE:
        return {"error": "SymPy non installato"}
    err = _validate_sympy_input(equation)
    if err:
        return {"error": err}
    try:
        var = symbols(variable)
        equation_stripped = equation.strip()
        # Rileva sistemi lineari: [eq1, eq2, ...] o (eq1, eq2, ...)
        if (equation_stripped.startswith("[") or equation_stripped.startswith("(")) and "=" in equation_stripped:
            inner = equation_stripped.strip("[]() ")
            # Usa sympy's solve con le equazioni già parsate
            eq_list = []
            for part in _split_eq_string(inner):
                if "=" in part:
                    left, right = part.split("=", 1)
                    eq_list.append(sympify(left.strip()) - sympify(right.strip()))
                else:
                    eq_list.append(sympify(part))
            vars_found = sorted(_extract_symbols_from_eqs(equation_stripped))
            if not vars_found:
                vars_found = [variable]
            solutions = _sympy_solve(eq_list, vars_found)
            return {
                "type": "system",
                "equations": equation_stripped,
                "variables": vars_found,
                "solution": {str(k): str(v) for k, v in solutions.items()},
                "numeric_solution": {
                    str(k): float(v.evalf()) if hasattr(v, 'evalf') else float(v)
                    for k, v in solutions.items()
                },
            }
        # Equazione singola con "="
        if "=" in equation_stripped:
            left, right = equation_stripped.split("=", 1)
            eq = sympify(left.strip()) - sympify(right.strip())
        else:
            eq = sympify(equation_stripped)
        solutions = _sympy_solve(eq, var)
        numeric = []
        for s in solutions:
            try:
                numeric.append(float(s.evalf()))
            except (TypeError, ValueError):
                numeric.append(float(s) if isinstance(s, (int, float, complex)) else str(s))
        return {
            "equation": equation_stripped,
            "variable": variable,
            "solutions": [str(s) for s in solutions],
            "numeric_solutions": numeric,
            "count": len(solutions),
        }
    except Exception as e:
        return {"error": f"Errore nella risoluzione: {str(e)}"}


def _split_eq_string(inner: str) -> list[str]:
    """Splits a string like 'x + y = 5, x - y = 1' respecting parentheses."""
    parts = []
    depth = 0
    current = ""
    for ch in inner:
        if ch in ("(", "["):
            depth += 1
            current += ch
        elif ch in (")", "]"):
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


def _extract_symbols_from_eqs(eq_str: str) -> list[str]:
    """Extracts variable names from an equation string."""
    cleaned = eq_str.replace("=", " ").replace("+", " ").replace("-", " ")
    cleaned = cleaned.replace("*", " ").replace("/", " ").replace("^", " ")
    cleaned = cleaned.replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ")
    tokens = set(cleaned.split())
    return sorted([s for s in tokens if s and s[0].isalpha() and not s.isdigit()])


@mcp_server.tool()
async def differentiate(expression: str, variable: str = "x", order: int = 1) -> dict:
    """Calcola la derivata di un'espressione simbolica.

    Args:
        expression: Espressione. Esempi: "x**3 + 2*x", "sin(x)*exp(x)"
        variable: Variabile di derivazione (default: "x")
        order: Ordine della derivata (default: 1, max: 10)
    """
    if not SYMPY_AVAILABLE:
        return {"error": "SymPy non installato"}
    err = _validate_sympy_input(expression)
    if err:
        return {"error": err}
    try:
        var = symbols(variable)
        expr = sympify(expression)
        result = _sympy_diff(expr, var, order)
        simp = _sympy_simplify(result)
        return {
            "expression": expression,
            "variable": variable,
            "order": order,
            "derivative": str(result),
            "simplified": str(simp),
            "result_latex": result.latex() if hasattr(result, 'latex') else str(result),
        }
    except Exception as e:
        return {"error": f"Errore nella derivazione: {str(e)}"}


@mcp_server.tool()
async def integrate(expression: str, variable: str = "x", lower_bound: float | None = None, upper_bound: float | None = None) -> dict:
    """Calcola integrali indefiniti o definiti.

    Args:
        expression: Espressione. Esempi: "x**2", "sin(x)", "1/x"
        variable: Variabile (default: "x")
        lower_bound: Limite inferiore (opzionale)
        upper_bound: Limite superiore (opzionale)
    """
    if not SYMPY_AVAILABLE:
        return {"error": "SymPy non installato"}
    err = _validate_sympy_input(expression)
    if err:
        return {"error": err}
    try:
        var = symbols(variable)
        expr = sympify(expression)
        if lower_bound is not None and upper_bound is not None:
            lb = sympify(str(lower_bound))
            ub = sympify(str(upper_bound))
            result = _sympy_integrate(expr, (var, lb, ub))
            numeric = float(result.evalf()) if hasattr(result, 'evalf') else float(result)
            return {
                "expression": expression,
                "variable": variable,
                "type": "definite",
                "bounds": [lower_bound, upper_bound],
                "result": str(result),
                "result_latex": result.latex() if hasattr(result, 'latex') else str(result),
                "numeric": numeric,
            }
        else:
            result = _sympy_integrate(expr, var)
            simp = _sympy_simplify(result)
            return {
                "expression": expression,
                "variable": variable,
                "type": "indefinite",
                "result": str(result),
                "simplified": str(simp),
                "result_latex": result.latex() if hasattr(result, 'latex') else str(result),
            }
    except Exception as e:
        return {"error": f"Errore nell'integrazione: {str(e)}"}


@mcp_server.tool()
async def limit_func(expression: str, variable: str = "x", point: float | None = None, direction: str = "both") -> dict:
    """Calcola il limite di una funzione.

    Args:
        expression: Espressione. Esempi: "sin(x)/x", "(1 + 1/x)**x"
        variable: Variabile (default: "x")
        point: Punto di avvicinamento (default: 0). Usa "oo" per infinito.
        direction: "left", "right" o "both" (default: "both")
    """
    if not SYMPY_AVAILABLE:
        return {"error": "SymPy non installato"}
    err = _validate_sympy_input(expression)
    if err:
        return {"error": err}
    try:
        var = symbols(variable)
        expr = sympify(expression)
        point_val = sympify(str(point)) if point is not None else 0
        if direction == "left":
            result = sympy_limit(expr, var, point_val, dir="-")
        elif direction == "right":
            result = sympy_limit(expr, var, point_val, dir="+")
        else:
            result = sympy_limit(expr, var, point_val)
        try:
            numeric = float(result.evalf()) if hasattr(result, 'evalf') else float(result)
        except (TypeError, ValueError):
            numeric = str(result)
        return {
            "expression": expression,
            "variable": variable,
            "point": str(point),
            "direction": direction,
            "limit": str(result),
            "limit_latex": result.latex() if hasattr(result, 'latex') else str(result),
            "numeric": numeric,
        }
    except Exception as e:
        return {"error": f"Errore nel calcolo del limite: {str(e)}"}


@mcp_server.tool()
async def simplify_expr(expression: str) -> dict:
    """Semplifica un'espressione simbolica.

    Args:
        expression: Espressione. Esempi: "x**2 + 2*x + x**2", "sin(x)**2 + cos(x)**2", "(x**2 - 1)/(x - 1)"
    """
    if not SYMPY_AVAILABLE:
        return {"error": "SymPy non installato"}
    err = _validate_sympy_input(expression)
    if err:
        return {"error": err}
    try:
        expr = sympify(expression)
        simp = _sympy_simplify(expr)
        factored = str(_sympy_factor(expr))
        expanded = str(_sympy_expand(expr))
        return {
            "expression": expression,
            "simplified": str(simp),
            "simplified_latex": simp.latex() if hasattr(simp, 'latex') else str(simp),
            "factored": factored,
            "expanded": expanded,
        }
    except Exception as e:
        return {"error": f"Errore nella semplificazione: {str(e)}"}


@mcp_server.tool()
async def symbolic_calculate(expression: str, variables: str | None = None) -> dict:
    """Calcoli simbolici e numerici con valutazione.

    Args:
        expression: Espressione matematica. Esempi: "sqrt(2) + pi", "factorial(10)"
        variables: Variabili in formato JSON (opzionale). Esempio: '{"a": 5, "b": 3}'
    """
    if not SYMPY_AVAILABLE:
        return {"error": "SymPy non installato"}
    err = _validate_sympy_input(expression)
    if err:
        return {"error": err}
    try:
        local_vars = {
            "pi": pi, "E": E, "oo": oo, "I": I,
            "sqrt": sqrt, "exp": exp, "log": log,
            "sin": sin, "cos": cos, "tan": tan,
            "asin": asin, "acos": acos, "atan": atan,
            "sinh": sinh, "cosh": cosh, "tanh": tanh,
            "factorial": factorial, "binomial": binomial, "abs": Abs, "ln": log,
            "__builtins__": {},
        }
        if variables:
            vars_dict = json.loads(variables)
            local_vars.update(vars_dict)
        result = sympify(expression, locals=local_vars)
        numeric = float(result.evalf()) if hasattr(result, 'evalf') else float(result)
        return {
            "expression": expression,
            "symbolic_result": str(result),
            "symbolic_latex": result.latex() if hasattr(result, 'latex') else str(result),
            "numeric": numeric,
        }
    except Exception as e:
        return {"error": f"Errore nel calcolo: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════
# CALCOLO NUMERICO (NumPy) — AST SAFE EVAL
# ═══════════════════════════════════════════════════════════════════════════

@mcp_server.tool()
async def numerical_calculate(expression: str, variables: str | None = None) -> dict:
    """Calcoli numerici complessi con NumPy (AST-safe evaluation).

    Args:
        expression: Espressione con prefisso np. Esempi: "np.sqrt(2) + np.exp(1)"
        variables: Variabili in formato JSON (opzionale).
    """
    if not NUMPY_AVAILABLE:
        return {"error": "NumPy non installato"}
    try:
        result = _safe_numpy_eval(expression)
        return {"expression": expression, "result": result, "dtype": str(type(result).__name__)}
    except Exception as e:
        return {"error": f"Errore nel calcolo numerico: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════
# OPERAZIONI MATRICIALI (NumPy + SciPy)
# ═══════════════════════════════════════════════════════════════════════════

@mcp_server.tool()
async def matrix_operations(matrix_str: str, operation: str = "det") -> dict:
    """Operazioni matriciali con NumPy/SciPy.

    Args:
        matrix_str: Matrice in formato JSON. Esempio: "[[1,2],[3,4]]"
        operation: "det", "eigenvalues", "eigenvectors", "inverse", "svd",
                   "transpose", "rank", "trace", "norm", "condition"
    """
    if not NUMPY_AVAILABLE or not SCIPY_AVAILABLE:
        return {"error": "NumPy/SciPy non installati"}
    try:
        matrix = np.array(json.loads(matrix_str), dtype=float)
        rows, cols = matrix.shape
        def _clean(v):
            if isinstance(v, (np.floating, np.integer)): return float(v)
            if isinstance(v, np.complexfloating): return {"real": float(v.real), "imag": float(v.imag)}
            if isinstance(v, np.ndarray): return v.tolist()
            return float(v) if hasattr(v, '__float__') else str(v)
        result = {"operation": operation, "matrix": matrix.tolist(), "shape": [rows, cols]}
        if operation == "det":
            result["determinant"] = _clean(np.linalg.det(matrix))
        elif operation == "eigenvalues":
            result["eigenvalues"] = [_clean(v) for v in np.linalg.eigvals(matrix)]
        elif operation == "eigenvectors":
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            result["eigenvalues"] = [_clean(v) for v in eigenvalues]
            result["eigenvectors"] = eigenvectors.tolist()
        elif operation == "inverse":
            result["inverse"] = np.linalg.inv(matrix).tolist()
        elif operation == "svd":
            U, s, Vt = np.linalg.svd(matrix)
            result["U"], result["singular_values"], result["Vt"] = U.tolist(), s.tolist(), Vt.tolist()
        elif operation == "transpose":
            result["transposed"] = matrix.T.tolist()
        elif operation == "rank":
            result["rank"] = int(np.linalg.matrix_rank(matrix))
        elif operation == "trace":
            result["trace"] = _clean(np.trace(matrix))
        elif operation == "norm":
            result["frobenius"] = _clean(np.linalg.norm(matrix, 'fro'))
            result["spectral"] = _clean(np.linalg.norm(matrix, 2))
        elif operation == "condition":
            result["condition_number"] = _clean(np.linalg.cond(matrix))
        else:
            return {"error": f"Operazione non valida: {operation}",
                "supported": ["det", "eigenvalues", "eigenvectors", "inverse", "svd", "transpose", "rank", "trace", "norm", "condition"]}
        return result
    except Exception as e:
        return {"error": f"Errore matriciale: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICA (NumPy + SciPy)
# ═══════════════════════════════════════════════════════════════════════════

@mcp_server.tool()
async def statistics(data: str, operation: str = "full", confidence: float = 0.95) -> dict:
    """Analisi statistica descrittiva con NumPy/SciPy.

    Args:
        data: Dati in formato JSON. Esempio: "[1,2,3,4,5]"
        operation: "full", "descriptive", "correlation", "normality", "regression", "anomaly"
        confidence: Livello di confidenza (default: 0.95)
    """
    if not NUMPY_AVAILABLE or not SCIPY_AVAILABLE:
        return {"error": "NumPy/SciPy non installati"}
    try:
        data_list = json.loads(data)
        if isinstance(data_list[0], list):
            arr = np.array(data_list, dtype=float)
            multi_variate = True
            n_vars = arr.shape[1] if arr.ndim > 1 else 1
        else:
            arr = np.array(data_list, dtype=float).flatten()
            multi_variate = False
            n_vars = 1
        result = {"operation": operation, "data_count": len(arr), "data_type": "multivariate" if multi_variate else "univariate", "n_variables": n_vars}
        if operation in ("full", "descriptive"):
            result["mean"] = float(np.mean(arr))
            result["median"] = float(np.median(arr))
            result["std"] = float(np.std(arr, ddof=1))
            result["variance"] = float(np.var(arr, ddof=1))
            result["min"] = float(np.min(arr))
            result["max"] = float(np.max(arr))
            result["sum"] = float(np.sum(arr))
            result["range"] = float(np.ptp(arr))
            result["quartile_25"] = float(np.percentile(arr, 25))
            result["quartile_75"] = float(np.percentile(arr, 75))
            result["quartile_interquartile"] = float(np.percentile(arr, 75) - np.percentile(arr, 25))
            result["skewness"] = float(scipy_stats.skew(arr))
            result["kurtosis"] = float(scipy_stats.kurtosis(arr))
        if operation in ("full", "correlation"):
            if multi_variate:
                result["correlation_matrix"] = np.corrcoef(arr).tolist()
                result["covariance_matrix"] = np.cov(arr).tolist()
                result["variable_names"] = [f"x{i+1}" for i in range(n_vars)]
        if operation in ("full", "normality"):
            if 8 <= len(arr) <= 5000:
                stat, p_value = scipy_stats.shapiro(arr)
                result["normality_test"] = {"test": "Shapiro-Wilk", "statistic": float(stat), "p_value": float(p_value), "is_normal": bool(p_value > 0.05)}
            elif len(arr) < 8:
                result["normality_test"] = {"note": "Shapiro-Wilk richiede almeno 8 osservazioni", "count": len(arr)}
        if operation in ("full", "anomaly"):
            if not multi_variate:
                z_scores = np.abs(scipy_stats.zscore(arr))
                anomalies = np.where(z_scores > 2.0)[0]
                result["anomaly_detection"] = {"method": "z-score", "threshold": 2.0, "anomaly_count": len(anomalies),
                    "anomaly_values": arr[anomalies].tolist(), "anomaly_indices": anomalies.tolist(), "z_scores": z_scores.tolist()}
        if operation in ("full", "regression"):
            if multi_variate and len(arr[0]) >= 2:
                x_data, y_data = arr[:, 0], arr[:, 1]
            elif not multi_variate and len(arr) >= 2:
                x_data = np.arange(len(arr), dtype=float)
                y_data = arr
            else:
                return {"error": "Regressione richiede almeno 2 dati"}
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x_data, y_data)
            result["regression"] = {"slope": float(slope), "intercept": float(intercept), "r_value": float(r_value),
                "r_squared": float(r_value ** 2), "p_value": float(p_value), "std_err": float(std_err),
                "equation": f"y = {slope:.6f}x + {intercept:.6f}"}
        return result
    except Exception as e:
        return {"error": f"Errore statistico: {str(e)}"}


# ═══════════════════════════════════════════════════════════════════════════
# AVVIO
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    print(f"🔢 Hermes MCP Math Server v2.2.0", file=sys.stderr)
    print(f"   Transport: {TRANSPORT}", file=sys.stderr)
    print(f"   SymPy: {'✓' if SYMPY_AVAILABLE else '✗'}", file=sys.stderr)
    print(f"   NumPy: {'✓' if NUMPY_AVAILABLE else '✗'}", file=sys.stderr)
    print(f"   SciPy: {'✓' if SCIPY_AVAILABLE else '✗'}", file=sys.stderr)
    if TRANSPORT == "stdio":
        print("\nRunning in STDIO mode...", file=sys.stderr)
        await mcp_server.run_stdio_async()
    elif TRANSPORT == "http":
        port = int(os.environ.get("HERMES_MCP_PORT", "18762"))
        if FASTMCP_AVAILABLE:
            print(f"\nRunning in HTTP mode on :{port}...", file=sys.stderr)
            from starlette.middleware.cors import CORSMiddleware
            mcp_app = mcp_server.streamable_http_app()
            cors_origins_list = CORS_ORIGINS if CORS_ORIGINS else ["http://localhost"]
            cors_app = CORSMiddleware(
                app=mcp_app, allow_origins=cors_origins_list, allow_methods=["POST", "OPTIONS"],
                allow_headers=["*"], expose_headers=["Mcp-Session-Id", "Cache-Control", "Content-Disposition"],
            )
            import uvicorn
            config = uvicorn.Config(cors_app, host="0.0.0.0", port=port, log_level="warning")
            server = uvicorn.Server(config)
            _shutdown_event = asyncio.Event()
            def _on_signal(_sig, _frame):
                print("\nShutting down...", file=sys.stderr)
                _shutdown_event.set()
            sig_mod.signal(sig_mod.SIGINT, _on_signal)
            sig_mod.signal(sig_mod.SIGTERM, _on_signal)
            try:
                await server.serve()
            except SystemExit:
                pass
            if _shutdown_event.is_set():
                print("Shutting down...", file=sys.stderr)
        else:
            print("\nERROR: FastMCP HTTP richiede 'mcp[serve]'.", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())
