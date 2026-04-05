from sympy import (
    symbols, sympify, sin, cos, tan, cot, sec, csc,
    sinh, cosh, tanh, coth, sech, csch, exp, log,
    diff, solve, simplify, factor, cancel, Symbol,
    Add, Mul, Pow, Wild, oo, zoo, nan, Integer, expand
)

# ── symbols ──────────────────────────────────────────────────────────────────
x  = symbols('x')
y  = symbols('y')
Dy = symbols('DY')        # dy/dx  placeholder
D2y= symbols('D2Y')       # d²y/dx²placeholder

allowed_funcs = {
    'sin': sin, 'cos': cos, 'tan': tan, 'cot': cot, 'sec': sec, 'csc': csc,
    'sinh': sinh, 'cosh': cosh, 'tanh': tanh, 'coth': coth,
    'sech': sech, 'csch': csch,
    'exp': exp, 'log': log,
}

# ── pre-processing ────────────────────────────────────────────────────────────
def preprocess(eq_str: str) -> str:
    s = eq_str.strip()
    # Replace dy/dx and d2y/dx2 BEFORE lowercasing to preserve case in placeholders
    # Support both cases
    import re
    s = re.sub(r'd2y/dx2', 'D2Y', s, flags=re.IGNORECASE)
    s = re.sub(r'dy/dx',   'DY',  s, flags=re.IGNORECASE)
    # Now lowercase everything except our placeholders
    s = s.lower()
    s = s.replace("d2y",  "D2Y")   # restore if any remain after lower
    s = s.replace("dy",   "DY")    # restore if any remain after lower
    # Handle = sign
    if "=" in s:
        idx = s.index("=")
        lhs = s[:idx]
        rhs = s[idx+1:]
        s = lhs + "-(" + rhs + ")"
    s = s.replace("^", "**")
    return s

# ── order ─────────────────────────────────────────────────────────────────────
def classify_order(eq) -> str:
    if eq.has(D2y):
        return "Second Order"
    if eq.has(Dy):
        return "First Order"
    return "Not a differential equation"

# ── helpers ───────────────────────────────────────────────────────────────────
def _safe(expr):
    """Return None if expr is nan / zoo / oo, else the expr."""
    if expr in (nan, zoo, oo, -oo):
        return None
    return expr

def _is_only_x(expr) -> bool:
    """True if expression contains x but NOT y (or is a constant)."""
    return not expr.has(y)

def _is_only_y(expr) -> bool:
    """True if expression contains y but NOT x (or is a constant)."""
    return not expr.has(x)

# ── exact / non-exact ────────────────────────────────────────────────────────
def parse_MN_from_string(eq_str: str):
    """
    Parse an equation of the form:
        M(x,y)*dx + N(x,y)*dy = 0
    or  M(x,y)*dx + N(x,y)*dy
    Returns (M, N) as sympy expressions, or raises ValueError.
    """
    import re
    s = eq_str.strip().lower()
    s = s.replace("^", "**")

    # Remove '= 0' or '=0' at the end if present
    s = re.sub(r'=\s*0\s*$', '', s).strip()

    # We need to split on 'dx' and 'dy' terms.
    # Strategy: find the coefficient of dx and dy.
    # We look for patterns like:  <expr>*dx  or  dx*<expr>  and same for dy

    dx_sym = symbols('dx')
    dy_sym = symbols('dy')

    # Parse the whole expression treating dx and dy as symbols
    try:
        expr = sympify(s, locals={
            **allowed_funcs,
            'x': x, 'y': y,
            'dx': dx_sym, 'dy': dy_sym,
        })
    except Exception as e:
        raise ValueError(f"Cannot parse expression: {e}")

    expr = expand(expr)

    M = expr.coeff(dx_sym)
    N = expr.coeff(dy_sym)

    if M == 0 and N == 0:
        raise ValueError("No dx or dy terms found")

    return M, N


def classify_exact_MN(M, N) -> str:
    """
    Given M and N from  M dx + N dy = 0,
    check if ∂M/∂y = ∂N/∂x  (Exact condition).
    """
    try:
        My = simplify(diff(M, y))
        Nx = simplify(diff(N, x))
        if simplify(My - Nx) == 0:
            return "Exact"
        return "Non-Exact"
    except Exception as e:
        return f"Error: {e}"


def classify_exact(eq_str: str) -> str:
    """
    Full exact classifier. Accepts the raw input string in the form:
        M(x,y)*dx + N(x,y)*dy = 0
    """
    try:
        M, N = parse_MN_from_string(eq_str)
        return classify_exact_MN(M, N)
    except ValueError as e:
        return f"Cannot determine ({e})"
    except Exception as e:
        return f"Error: {e}"

# ── first-order type ──────────────────────────────────────────────────────────
def classify_type(eq) -> str:
    from sympy import Poly, cancel, expand

    # ── solve for dy/dx ──────────────────────────────────────────────────────
    try:
        sol = solve(eq, Dy)
    except Exception:
        return "Cannot solve for dy/dx"

    if not sol:
        return "Cannot classify (no solution for dy/dx)"

    f = simplify(sol[0])

    if f in (nan, zoo, oo, -oo):
        return "Cannot classify (degenerate)"

    types = []

    # ── try to express f as a polynomial in y ────────────────────────────────
    try:
        p   = Poly(expand(f), y)
        deg = p.degree()
        pc  = {m[0]: c for m, c in zip(p.monoms(), p.coeffs())}
        all_x = all(_is_only_x(c) for c in pc.values())
    except Exception:
        p = None; deg = None; pc = {}; all_x = False

    # ─────────────────────────────────────────────────────────────────────────
    # LINEAR :  f = a(x)·y + b(x)   with b(x) ≠ 0
    #           (if b(x) = 0  it is also separable → handled below)
    # ─────────────────────────────────────────────────────────────────────────
    is_linear = False
    if all_x and deg is not None and deg == 1:
        a = pc.get(1, Integer(0))
        b = pc.get(0, Integer(0))
        if _is_only_x(a) and _is_only_x(b):
            is_linear = True
            if simplify(b) != 0:
                types.append("Linear")   # true linear (non-homogeneous)

    # ─────────────────────────────────────────────────────────────────────────
    # SEPARABLE :  f = g(x) · h(y)
    #   - If linear with b=0  →  dy/dx = a(x)·y  which IS separable
    #   - Otherwise use numeric probe to detect separability
    # ─────────────────────────────────────────────────────────────────────────
    if "Linear" not in types:   # only look for separable if NOT linear+nonhomog
        sep_found = False

        # Case A: linear with b=0  →  a(x)*y  is separable
        if is_linear and simplify(pc.get(0, Integer(0))) == 0:
            sep_found = True

        # Case B: numeric-probe factorability test
        if not sep_found:
            try:
                for cy in [2, 3, 5]:
                    gx = f.subs(y, cy)
                    if gx == 0 or not _safe(gx):
                        continue
                    ratio = simplify(cancel(f / gx))
                    if not ratio.has(x) and ratio.has(y):
                        # confirm other direction
                        for cx in [2, 3]:
                            hy = f.subs(x, cx)
                            if hy == 0 or not _safe(hy):
                                continue
                            ratio2 = simplify(cancel(f / hy))
                            if not ratio2.has(y):
                                sep_found = True
                                break
                    if sep_found:
                        break
            except Exception:
                pass

        if sep_found:
            types.append("Separable")

    # ─────────────────────────────────────────────────────────────────────────
    # BERNOULLI :  f = a(x)·y + b(x)·y^n   (n ≠ 0, 1)
    # ─────────────────────────────────────────────────────────────────────────
    if "Linear" not in types and "Separable" not in types:
        try:
            if all_x and deg is not None and deg >= 2:
                powers = set(pc.keys())
                non_one = powers - {1}        # powers other than y^1
                non_one_non_zero = non_one - {0}   # also exclude constant
                if len(non_one_non_zero) == 1:
                    n_val = list(non_one_non_zero)[0]
                    # remaining powers must only be 0 and/or 1
                    remaining = powers - {n_val}
                    if remaining <= {0, 1}:
                        types.append(f"Bernoulli (n={n_val})")
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # FALLBACK
    # ─────────────────────────────────────────────────────────────────────────
    if not types:
        types.append("Non-Linear / Special Form")

    return ", ".join(types)

# ── detect input form ────────────────────────────────────────────────────────
def _is_exact_form(eq_str: str) -> bool:
    """Return True if input looks like  M dx + N dy = 0  form."""
    import re
    s = eq_str.lower()
    return bool(re.search(r'\bdx\b', s) and re.search(r'\bdy\b', s))

# ── main loop ─────────────────────────────────────────────────────────────────
def run():
    print("ODE Classifier")
    print("  - For dy/dx form:   e.g.  dy/dx + 2*y = x^2")
    print("  - For M dx+N dy=0:  e.g.  (2*x*y)*dx + (x^2+y^2)*dy = 0")

    while True:
        eq_str = input("\nEnter ODE: ").strip()
        if not eq_str:
            continue

        try:
            # ── Branch 1: M*dx + N*dy = 0  form ─────────────────────────────
            if _is_exact_form(eq_str):
                print(f"\n  Order : First Order")
                print(f"  Form  : M dx + N dy = 0")
                try:
                    M, N = parse_MN_from_string(eq_str)
                    print(f"  M(x,y) = {M}")
                    print(f"  N(x,y) = {N}")
                    result = classify_exact_MN(M, N)
                    if result == "Exact":
                        print(f"  Exact : ✓ Yes  (∂M/∂y = ∂N/∂x = {simplify(diff(M,y))})")
                    else:
                        print(f"  Exact : ✗ No   (∂M/∂y = {simplify(diff(M,y))}, ∂N/∂x = {simplify(diff(N,x))})")
                except Exception as e:
                    print(f"  Exact : Cannot determine ({e})")

            # ── Branch 2: dy/dx form ─────────────────────────────────────────
            else:
                processed = preprocess(eq_str)
                eq = sympify(processed, locals={
                    **allowed_funcs,
                    'x': x, 'y': y, 'DY': Dy, 'D2Y': D2y,
                })

                order = classify_order(eq)
                print(f"\n  Order : {order}")

                if "first" in order.lower():
                    print(f"  Type  : {classify_type(eq)}")

                elif "second" in order.lower():
                    print("  Type  : Second-Order (further analysis needed)")
                else:
                    print("  Type  : N/A")

        except Exception as e:
            print(f"  !! Parse / evaluation error: {e}")

        again = input("\nAnother equation? (y/n): ").strip().lower()
        if again != 'y':
            print("Goodbye!")
            break

if __name__ == "__main__":
    run()
