## Author: Jens Stücker (University of Vienna), 2025
## taken from https://github.com/jstuecker/jax-mysteries
import jax, jax.numpy as jnp
from jaxlib import xla_client as xc
from graphviz import Source
from IPython.display import SVG, display, Image, HTML
import re
import numpy as np

def bytes_str(bytes):
    if abs(bytes) < 1024:
        return f"{bytes} B"
    elif abs(bytes) < 1024**2:
        return f"{bytes / 1024:.1f} kB"
    elif abs(bytes) < 1024**3:
        return f"{bytes / 1024**2:.1f} MB"
    else:
        return f"{bytes / 1024**3:.1f} GB"

def print_memory_usage(flowered, fcompiled=None, show_host_mem=False):
    if fcompiled is None:
        fcompiled = flowered.compile()
    m = fcompiled.memory_analysis()

    # if m.generated_code_size_in_bytes > 1024*1024:
    #     print("Warning! We have constant folding!")

    print(f"const : {bytes_str(folded_constants_bytes(flowered))}")
    print(f"code  : {bytes_str(m.generated_code_size_in_bytes )}")
    print(f"temp  : {bytes_str(m.temp_size_in_bytes)}")
    print(f"arg   : {bytes_str(m.argument_size_in_bytes)}")
    print(f"output: {bytes_str(m.output_size_in_bytes)}")
    print(f"alias : {bytes_str(-m.alias_size_in_bytes)}")
    print(f"peak  : {bytes_str(m.peak_memory_in_bytes )}")

    if show_host_mem:
        print(f"host code : {bytes_str(m.host_generated_code_size_in_bytes )}")
        print(f"host temp : {bytes_str(m.host_temp_size_in_bytes)}")
        print(f"host arg  : {bytes_str(m.host_argument_size_in_bytes)}")
        print(f"host output : {bytes_str(m.host_output_size_in_bytes)}")
        print(f"host alias : {bytes_str(m.host_alias_size_in_bytes)}")

def hlo_to_svg_text(hlo_text: str, title: str = None):
    # Parse HLO text -> XLA HloModule, then emit DOT and render to SVG
    mod = xc._xla.hlo_module_from_text(hlo_text)        # private API
    dot = xc._xla.hlo_module_to_dot_graph(mod)          # private API
    svg_bytes = Source(dot).pipe(format="svg")

    # svg_bytes is bytes; convert to text so we can optionally inject a <title>
    if isinstance(svg_bytes, bytes):
        svg_text = svg_bytes.decode("utf-8")
    else:
        svg_text = str(svg_bytes)

    if title:
        title_fragment = (
            f"<title>{title}</title>"
            f"<g id=\"hlo_title\">"
            f"<text x=\"22\" y=\"20\" font-family=\"sans-serif\" "
            f"font-size=\"16\" font-weight=\"bold\" fill=\"#1976d2\">{title}</text>"
            f"</g>"
        )
        if re.search(r'</svg\s*>', svg_text, flags=re.IGNORECASE):
            svg_text = re.sub(r'</svg\s*>', title_fragment + '</svg>', svg_text, count=1, flags=re.IGNORECASE)
        else:
            svg_text = title_fragment + svg_text

    return svg_text

def resize_svg(svg, width=400):
    """Return an SVG string with width fixed to `width` and height auto."""
    if hasattr(svg, "data"):          # IPython.display.SVG
        svg_text = svg.data
    elif isinstance(svg, bytes):
        svg_text = svg.decode("utf-8")
    else:
        svg_text = str(svg)

    # remove any existing width/height on the <svg> tag
    svg_text = re.sub(r'(<svg[^>]*?)\swidth="[^"]*"', r'\1', svg_text, count=1)
    svg_text = re.sub(r'(<svg[^>]*?)\sheight="[^"]*"', r'\1', svg_text, count=1)
    # add our width; let height be computed from viewBox
    svg_text = re.sub(r'<svg', f'<svg width="{width}px"', svg_text, count=1)
    return SVG(svg_text)

def show_hlo_info(f, *args, mode="mem_post", width=400, save=False, show_host_mem=False, **kwargs):
    lo = f.lower(*args, **kwargs)
    title = f.__name__

    comp = lo.compile()

    if "mem" in mode:
        print(f"--------  Memory usage of {title}  ---------")
        print_memory_usage(lo, show_host_mem=show_host_mem)
    if "pre" in mode:
        pre_hlo  = lo.as_text(dialect="hlo")
        svg = hlo_to_svg_text(pre_hlo, title=f"{title} (pre)")
        if save:
            with open(f"{title}_pre.svg", "w") as f:
                f.write(svg)
        else:
            display(resize_svg(svg, width=width))
    if "post" in mode:
        m = comp.memory_analysis()
        mem_fac = m.temp_size_in_bytes / np.maximum(m.argument_size_in_bytes, 1)
        title_post = f"{title} (tmp: {bytes_str(m.temp_size_in_bytes)}, {mem_fac:.2g}x)"

        post_hlo = comp.as_text()
        svg = hlo_to_svg_text(post_hlo, title=title_post)
        if save:
            with open(f"{title}_post.svg", "w") as f:
                f.write(svg)
        else:
            display(resize_svg(svg, width=width))

def token_to_jnp_dtype(tok: str):
    """Map MLIR/StableHLO tokens to JAX/NumPy dtypes."""
    # booleans
    if tok == "i1":
        return jnp.bool_

    # complex
    if tok.startswith("complex<") and tok.endswith(">"):
        inner = tok[len("complex<"):-1]
        return {"f32": jnp.complex64, "f64": jnp.complex128}[inner]

    # integers: si*/ui*/i*
    m = re.fullmatch(r'(si|ui|i)(\d+)', tok)
    if m:
        kind, bits = m.groups()
        bits = int(bits)
        if kind == "ui":
            return jnp.dtype(f"uint{bits}")
        elif kind in ("si", "i"):
            return jnp.dtype(f"int{bits}")

    # fp8 & microscaling families
    FP_MAP = {
        "f8E3M4": jnp.float8_e3m4,
        "f8E4M3": jnp.float8_e4m3,
        "f8E4M3FN": jnp.float8_e4m3fn,
        "f8E4M3FNUZ": jnp.float8_e4m3fnuz,
        "f8E4M3B11FNUZ": jnp.float8_e4m3b11fnuz,
        "f8E5M2": jnp.float8_e5m2,
        "f8E5M2FNUZ": jnp.float8_e5m2fnuz,
        "f8E8M0FNU": jnp.float8_e8m0fnu,
        "f4E2M1FN": jnp.float4_e2m1fn,
        # "f6E2M3FN": jnp.float6_e2m3fn,
        # "f6E3M2FN": jnp.float6_e3m2fn,
    }
    if tok in FP_MAP:
        return FP_MAP[tok]

    # standard floats + bf16 + tf32
    if tok in {"bf16", "f16", "f32", "f64"}:
        return {"bf16": jnp.bfloat16, "f16": jnp.float16,
                "f32": jnp.float32, "f64": jnp.float64}[tok]
    if tok == "tf32":
        return jnp.float32  # StableHLO 'tf32' ⇒ closest array dtype is float32
    
    print(f"Warning: unknown dtype token {tok}. I'll assume float32 instead")
    return jnp.float32

def shape_dtype_to_struct(spec: str) -> jax.ShapeDtypeStruct:
    """
    Convert a captured 'shapexdtype' spec from tensor<...> into ShapeDtypeStruct.
    Examples:
      '131584x2xf32' -> shape=(131584, 2), dtype=float32
      'f32'          -> shape=(), dtype=float32 (scalar tensor)
    """
    m2 = re.match(
        r'^(?:(\d+(?:x\d+)*)x)?('
        r'complex<[^<>]+>|'
        r'i1|'
        r'(?:si|ui)(?:2|4|8|16|32|64)|'      # si*/ui*
        r'i(?:8|16|32|64)|'                  # legacy i*
        r'bf16|f16|f32|f64|tf32|'
        r'f(?:4E2M1FN|6E2M3FN|6E3M2FN|'
        r'8E3M4|8E4M3(?:B11FNUZ|FNUZ|FN)?|'
        r'8E5M2(?:FNUZ)?|8E8M0FNU)'
        r')$',
        spec
    )
    if not m2:
        raise ValueError(f"Unparsable tensor spec: {spec}")
    dims = tuple(map(int, m2.group(1).split('x'))) if m2.group(1) else ()
    dtype_tok = m2.group(2)
    return jax.ShapeDtypeStruct(shape=dims, dtype=token_to_jnp_dtype(dtype_tok))

PATTERN = re.compile(
    r'(?m)^\s*%cst(?:_\d+)?\s*=\s*stablehlo\.constant\b[^\n]*?:\s*tensor<((?:[^<>]|<[^<>]*>)+)>'
)
def detect_folded_constants(low):
    """Return a list of ShapeDtypeStruct for all %cst tensor types of a lowered jax function
    
    example: detect_folded_constants(jax.jit(f).lower(x))
    """
    shlo = low.compiler_ir(dialect="stablehlo")
    txt = shlo.operation.get_asm(large_elements_limit=16)
    return [shape_dtype_to_struct(m.group(1)) for m in PATTERN.finditer(txt)]

def folded_constants_bytes(low):
    """Return the total size in bytes of all folded constants in a lowered jax function
    
    example: folded_constants_bytes(jax.jit(f).lower(x))
    """
    consts = detect_folded_constants(low)
    return sum(c.size * c.dtype.itemsize for c in consts)