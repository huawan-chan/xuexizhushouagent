"""Agent 工具集：
  1) math_calculator        —— AST 白名单安全计算器(不 eval 任意代码)
  2) knowledge_base_search  —— Chroma 本地 PDF 知识库检索
  3) web_search             —— 网页搜索(可选：Tavily / DuckDuckGo)
"""
from __future__ import annotations

import ast
import math
import operator
import re
from typing import List, Optional

# =====================================================================
# 1) 数学计算器(安全实现)
# =====================================================================
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "log": math.log, "ln": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "abs": abs, "round": round,
    "floor": math.floor, "ceil": math.ceil,
    "degrees": math.degrees, "radians": math.radians,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}


def _normalize_expr(expr: str) -> str:
    """把常见全角/自然写法规整成 Python 表达式。"""
    e = expr.strip()
    e = e.replace("×", "*").replace("·", "*").replace("÷", "/")
    e = e.replace("−", "-").replace("—", "-").replace("^", "**")
    e = e.replace("π", "pi").replace("（", "(").replace("）", ")")
    e = e.replace("２", "2").replace("４", "4").replace("８", "8")  # 常见误输全角数字兜底
    e = e.replace("，", ",")
    return e


def _eval_node(node: ast.AST, depth: int = 0) -> float:
    """递归求值。只允许白名单内的语法结构。"""
    if depth > 200:
        raise ValueError("表达式嵌套过深，已拒绝计算。")
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, depth)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"不支持的常量类型: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTS:
            return _ALLOWED_CONSTS[node.id]
        raise ValueError(f"不允许的变量名: {node.id}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval_node(node.operand, depth + 1)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)
        op = _BIN_OPS[type(node.op)]
        if isinstance(node.op, ast.Pow):
            if abs(right) > 100 or abs(left) > 1e6 and right > 10:
                raise ValueError("指数过大，拒绝计算(防止数值溢出)。")
        try:
            val = op(left, right)
        except ZeroDivisionError:
            raise ValueError("除数不能为 0。")
        if isinstance(val, complex) or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            raise ValueError("结果无效(NaN / Infinity)。")
        return float(val)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError("只支持数学函数: sqrt/sin/cos/tan/log/ln/exp/abs/round/floor/ceil 等。")
        fname = node.func.id
        args = [_eval_node(a, depth + 1) for a in node.args]
        if len(args) == 0 or len(args) > 2:
            raise ValueError(f"函数 {fname} 参数数量不对。")
        if fname in ("sqrt", "log", "ln", "log10", "log2", "factorial") and not args:
            raise ValueError(f"函数 {fname} 缺少参数。")
        if fname == "factorial":
            if any(a < 0 or a != int(a) for a in args):
                raise ValueError("factorial 只接受非负整数。")
            return float(math.factorial(int(args[0])))
        return float(_ALLOWED_FUNCS[fname](*args))
    raise ValueError("表达式包含不支持的语法(仅允许数字 + - * / % ** 与数学函数)。")


def _fmt_result(x: float) -> str:
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    return s


def calculate(expression: str) -> str:
    """对数学表达式做安全求值，返回人类可读的结果文本(供 LLM 引用)。"""
    expr = _normalize_expr(expression)
    if not expr:
        raise ValueError("表达式为空。")
    tree = ast.parse(expr, mode="eval")
    value = _eval_node(tree)
    return f"{expression.strip()} = {_fmt_result(value)}"


# =====================================================================
# 2) RAG：本地 PDF 知识库检索
# =====================================================================
def knowledge_base_search(query: str, top_k: Optional[int] = None) -> str:
    """检索本地 PDF 知识库(Chroma)，返回带来源页码的片段。"""
    from rag import vector_store as vs

    if not vs.has_index():
        return (
            "本地知识库目前为空。请在项目根目录执行: python cli.py build-index\n"
            "(把 PDF 放进 data/pdfs 后再建库)。在知识库有内容之前，请不要编造资料内容。"
        )
    results = vs.similarity_search(query, top_k=top_k)
    if not results:
        return f"在本地知识库中没有找到与「{query}」相关的内容，请如实告知用户，不要编造。"
    lines = [f"在资料中找到以下与「{query}」相关的片段(共 {len(results)} 条):\n"]
    for doc, score in results:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "?")
        snippet = doc.page_content.strip().replace("\n", " ")
        if len(snippet) > 450:
            snippet = snippet[:450] + "…"
        lines.append(f"[来源:{source} / 第{page}页 / 相似度{score:.2f}]\n{snippet}\n")
    return "\n".join(lines)


# =====================================================================
# 3) 网页搜索(可选)：优先 Tavily，否则回退 DuckDuckGo
# =====================================================================
def _search_duckduckgo(query: str, results_n: int) -> List[dict]:
    """免费的 DuckDuckGo HTML 接口抓取(无需 key)。失败返回 []。"""
    import requests
    from bs4 import BeautifulSoup

    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    try:
        resp = requests.post(url, data={"q": query}, headers=headers, timeout=12)
        resp.raise_for_status()
    except Exception:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    out: List[dict] = []
    for res in soup.select(".result")[: results_n]:
        a = res.select_one(".result__a")
        sn = res.select_one(".result__snippet")
        if not a:
            continue
        title = a.get_text(" ", strip=True)
        href = a.get("href", "")
        # ddg 返回跳转链，解析出真实 url
        m = re.search(r"uddg=([^&]+)", href)
        real = requests.utils.unquote(m.group(1)) if m else href
        snippet = sn.get_text(" ", strip=True) if sn else ""
        out.append({"title": title, "url": real, "snippet": snippet})
    return out


def _search_tavily(query: str, api_key: str, results_n: int) -> List[dict]:
    import requests

    resp = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": api_key, "query": query, "max_results": results_n},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    out = []
    for r in data.get("results", [])[: results_n]:
        out.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": (r.get("content") or "")[:300],
            }
        )
    return out


def web_search(query: str) -> str:
    """联网搜索并整理摘要返回(未开启/失败会明确提示，绝不伪造结果)。"""
    from config import get_settings

    settings = get_settings()
    if not settings.web_search_enabled:
        return "网页搜索未开启(可设置 WEB_SEARCH_ENABLED=true)。请如实告知用户。"
    n = settings.web_search_results
    try:
        if settings.tavily_api_key:
            results = _search_tavily(query, settings.tavily_api_key, n)
        else:
            results = _search_duckduckgo(query, n)
    except Exception as exc:
        return f"网页搜索失败: {exc}。请如实告知用户本次未能联网。"
    if not results:
        return f"网页搜索没有返回「{query}」的结果，请如实告知用户。"
    lines = [f"以下是「{query}」的搜索结果(共 {len(results)} 条，请甄别使用):\n"]
    for r in results:
        lines.append(f"- {r['title']}\n  {r['url']}\n  {r['snippet']}\n")
    return "\n".join(lines)


# =====================================================================
# 组装 LangChain 工具(供 bind_tools / ToolNode 使用)
# =====================================================================
def get_tools(with_web: bool = True) -> list:
    from langchain_core.tools import tool

    @tool("math_calculator")
    def math_calculator_tool(expression: str) -> str:
        """数学计算器：对任意算术/数学表达式做精确计算并返回结果。
        支持 + - * / % ** ( ) 和函数 sqrt/sin/cos/tan/log/ln/exp/abs/round/floor/ceil/factorial，常数 pi/e。
        参数 expression 为纯数学表达式，例如: "2**10", "sqrt(9) + (3*4)", "sin(pi/2)"。
        需要数值时务必先调用本工具，把得到的结果写进回答。"""
        return calculate(expression)

    @tool("knowledge_base_search")
    def knowledge_base_tool(query: str) -> str:
        """在老师上传的本地 PDF 知识库(讲义/教材/资料)中做语义检索，返回带来源页码的相关段落。
        当学生问题涉及你拥有的本地资料、或用户要求“按讲义/课本讲”时调用。
        参数 query 用简短的检索式中文描述要找的内容，例如: "牛顿第二定律 F=ma 公式"。"""
        return knowledge_base_search(query)

    @tool("web_search")
    def web_search_tool(query: str) -> str:
        """联网搜索网页并返回摘要(标题+链接+简介)。用于需要最新/外部事实、时效信息、背景资料时。
        参数 query 为搜索关键词，例如: "2026 高考物理大纲变化"。"""
        return web_search(query)

    tools = [math_calculator_tool, knowledge_base_tool]
    if with_web:
        tools.append(web_search_tool)
    return tools


def get_tool_names(with_web: bool = True) -> List[str]:
    return [t.name for t in get_tools(with_web=with_web)]
