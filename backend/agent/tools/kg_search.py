"""kg_search 工具节点。

设计原则：复制 dify_code_node_kg_lite.py 的已验证逻辑到本文件，
原文件保留不动。HugeGraph 地址从 config 读取，社区报告通过统一检索门面读取。
"""
import json
import time
import re
import logging
import urllib.parse
import requests
from typing import Dict

from backend.config.config import config
from backend.storage.search_store import search_store

logger = logging.getLogger(__name__)

_HG = f"http://{config.HUGEGRAPH_HOST}:{config.HUGEGRAPH_PORT}"


def _quote_vid(vid):
    """HugeGraph CUSTOMIZE_STRING 顶点 ID 必须双引号包裹"""
    return urllib.parse.quote('"' + vid + '"')


def _pv(props, key, default=""):
    """取 HugeGraph 属性值，兼容 dict{value:xxx} 和直接 string"""
    v = props.get(key, default)
    if isinstance(v, dict):
        return v.get("value", default)
    return v if isinstance(v, str) else default


def _hg_vertices(kb_ids, limit=500, max_pages=3):
    """按 kb_ids 查 HugeGraph 顶点（分页拉取，最多 max_pages 页；多库循环合并）。

    注意：HugeGraph 的分页是 base64 游标（响应体 page 字段），不是页码数字，
    不能传 page=1/2/3（会 400）。首次请求不带 page，用响应里的游标翻页。
    读取的属性：name/entity_type/description/weight/source_type/doc_id。
    source_type 用于识别故障树顶点（source_type="fault_tree"）。
    """
    if isinstance(kb_ids, str):
        kb_ids = [kb_ids]
    out = {}
    for kb_id in (kb_ids or []):
        if not kb_id:
            continue
        props = json.dumps({"kb_id": kb_id})
        page_token = None
        for _ in range(max_pages):
            params = {"properties": props, "limit": limit}
            if page_token:
                params["page"] = page_token
            try:
                r = requests.get(f"{_HG}/graphs/{config.HUGEGRAPH_GRAPH_NAME}/graph/vertices",
                                 params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
            except Exception:
                break
            verts = data.get("vertices", []) or []
            for v in verts:
                p = v.get("properties", {})
                name = _pv(p, "name", "")
                if name and v["id"] not in out:
                    out[v["id"]] = {
                        "id": v["id"], "name": name,
                        "type": _pv(p, "entity_type", ""),
                        "desc": _pv(p, "description", ""),
                        "weight": _pv(p, "weight", 0),
                        "source_type": _pv(p, "source_type", ""),
                        "doc_id": _pv(p, "doc_id", ""),
                    }
            page_token = data.get("page") or ""
            if not verts or not page_token:
                break
    return out


def _hg_edges(vid, limit=20):
    """查某顶点的邻接边（BOTH 方向）"""
    r = requests.get(f"{_HG}/graphs/{config.HUGEGRAPH_GRAPH_NAME}/graph/edges",
                     params={"vertex_id": '"' + vid + '"', "direction": "BOTH", "limit": limit},
                     timeout=10)
    return r.json().get("edges", [])


def _search_reports(kb_ids, entity_names, query, tk):
    """通过 OpenSearch 门面查社区报告，不在工具层拼接 DSL/HTTP。"""
    if isinstance(kb_ids, str):
        kb_ids = [kb_ids]
    kb_ids = [k for k in (kb_ids or []) if k]
    if not kb_ids or (not entity_names and not query):
        return []
    try:
        hits = []
        for kb_id in kb_ids:
            hits.extend(search_store.search_community_reports(
                kb_id=kb_id,
                query=query or None,
                entity_names=entity_names[:20],
                top_k=tk,
            ))
        out = []
        for h in hits:
            if hasattr(h, "document"):
                s = h.document
                score = h.score
            else:
                s = h.get("document") or h.get("_source") or h
                score = h.get("score", h.get("_score", 0))
            rep = s.get("report", {})
            if not isinstance(rep, dict):
                rep = {"title": str(rep)[:80], "summary": str(rep)}
            findings = rep.get("findings", []) or []
            # 标准化 findings 结构
            findings_norm = []
            for f in findings:
                if isinstance(f, dict):
                    findings_norm.append({
                        "summary": (f.get("summary") or "")[:200],
                        "explanation": (f.get("explanation") or "")[:200],
                    })
                elif isinstance(f, str):
                    findings_norm.append({"summary": f[:200], "explanation": ""})
            out.append({
                "title": rep.get("title", "") or "未命名社区",
                "summary": rep.get("summary", "") or "",
                "rating": rep.get("rating"),
                "rating_explanation": rep.get("rating_explanation", ""),
                "findings": findings_norm,
                "community_id": s.get("community_id"),
                "entities": s.get("entities", []) or [],
                "doc_ids": s.get("doc_ids", []) or [],
                "score": score or 0,
            })
        out.sort(key=lambda item: item.get("score", 0), reverse=True)
        return out[:tk]
    except Exception:
        return []


def _match_entities(all_ents, query, entity_kws, type_kws, tk):
    """从全量顶点中匹配实体"""
    matched = {}
    q_lower = query.lower()
    for vid, e in all_ents.items():
        name = e["name"].lower()
        if any(k in name for k in entity_kws) or any(k in q_lower for k in [e["name"]]):
            matched[vid] = {**e, "match": "name"}
    if not matched:
        for vid, e in all_ents.items():
            if e["name"] and e["name"] in query:
                matched[vid] = {**e, "match": "name_in_query"}
    if type_kws:
        for vid, e in all_ents.items():
            et = e.get("type", "").lower()
            if any(k.lower() in et for k in type_kws):
                if vid not in matched:
                    matched[vid] = {**e, "match": "type"}
    return list(matched.values())[:tk * 2]


def _edge_to_rel(edge, vid, all_ents):
    """把一条 HugeGraph 边转成关系 dict（目标顶点必须在 all_ents 内，保证数据隔离）"""
    outv = edge.get("outV", "")
    inv = edge.get("inV", "")
    if outv != vid and inv != vid:
        return None
    tgt_id = inv if outv == vid else outv
    if tgt_id not in all_ents:
        return None
    p = edge.get("properties", {})
    return {
        "src": all_ents[vid]["name"],
        "tgt": all_ents[tgt_id]["name"],
        "desc": _pv(p, "description", ""),
        "type": _pv(p, "relation_type", ""),
        "weight": _pv(p, "weight", 0),
        "edge_label": edge.get("label", ""),
        "source_type": _pv(p, "source_type", ""),
        "gate_type": _pv(p, "gate_type", ""),
        "gate_group": _pv(p, "gate_group", ""),
        "tgt_id": tgt_id,
    }


def _expand_neighbors(seed_ents, all_ents, tk, depth=2, max_nodes=40, max_edges=60):
    """BFS 多跳展开邻接边（默认 2 跳）。

    升级说明：原实现只做 1 跳展开，导致"过热→冷却系统→水泵损坏"这类
    中间隔着组件的 2 跳因果链查不到；这里改为受限 BFS 逐层扩展。
    """
    visited = set(e["id"] for e in seed_ents)
    frontier = [e["id"] for e in seed_ents[:5]]
    nodes_seen = set(visited)
    rels = []
    seen_keys = set()

    for _ in range(depth):
        next_frontier = []
        for vid in frontier:
            if len(rels) >= max_edges or len(nodes_seen) >= max_nodes:
                break
            try:
                edges = _hg_edges(vid, tk * 4)
            except Exception:
                continue
            for edge in edges:
                rel = _edge_to_rel(edge, vid, all_ents)
                if not rel:
                    continue
                k = (rel["src"], rel["tgt"], rel["type"], rel["gate_type"])
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                rels.append(rel)
                if rel["tgt_id"] not in nodes_seen:
                    nodes_seen.add(rel["tgt_id"])
                    next_frontier.append(rel["tgt_id"])
        frontier = next_frontier[:10]
        if not frontier:
            break

    neighbors = [all_ents[t] for t in nodes_seen if t not in visited]
    return neighbors, rels


def _expand_fault_tree(seed_vids, all_ents, max_nodes=120):
    """BFS 沿 source_type=fault_tree 的边展开完整故障树因果链。

    从命中的故障树顶点出发，向上回溯到顶事件、向下延伸到所有底事件，
    返回 (故障树顶点列表, 因果链关系列表)。关系带 gate_type（AND/OR）。
    """
    frontier = list(seed_vids)
    visited = set(seed_vids)
    ft_rels = []
    seen_keys = set()

    while frontier:
        next_frontier = []
        for vid in frontier:
            if len(visited) >= max_nodes:
                break
            try:
                edges = _hg_edges(vid, 100)
            except Exception:
                continue
            for edge in edges:
                p = edge.get("properties", {})
                if _pv(p, "source_type", "") != "fault_tree":
                    continue
                outv = edge.get("outV", "")
                inv = edge.get("inV", "")
                if outv != vid and inv != vid:
                    continue
                tgt = inv if outv == vid else outv
                if tgt not in all_ents:
                    continue
                k = (all_ents[vid]["name"], all_ents[tgt]["name"], _pv(p, "gate_type", ""))
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                ft_rels.append({
                    "src": all_ents[vid]["name"],
                    "tgt": all_ents[tgt]["name"],
                    "gate_type": _pv(p, "gate_type", ""),
                    "gate_group": _pv(p, "gate_group", ""),
                    "edge_label": edge.get("label", ""),
                    "desc": _pv(p, "description", ""),
                })
                if tgt not in visited:
                    visited.add(tgt)
                    next_frontier.append(tgt)
        frontier = next_frontier[:30]

    ft_ents = [all_ents[v] for v in visited]
    return ft_ents, ft_rels


def _build_summary(ents, rels, reports, query, ek, tk_kws, ft_ents=None, ft_rels=None):
    """构建给 LLM 的检索结果摘要。

    顺序设计：故障树因果链（若命中）→ 社区报告 → 相关实体 → 关系。
    故障树因果链是诊断类问题（"为什么/怎么回事/原因"）最直接的回答来源，
    社区报告是高层抽象总结，两者优先；一旦总长超限被截断，优先保留前面的内容。
    """
    ft_ents = ft_ents or []
    ft_rels = ft_rels or []
    L = [f"【用户问题】{query}\n"]
    if ek:
        L.append(f"【实体关键词】{', '.join(ek)}")
    if tk_kws:
        L.append(f"【类型关键词】{', '.join(tk_kws)}")

    # ── 故障树因果链（诊断类问题的核心，放最前面）──
    if ft_rels:
        L.append(f"\n【故障树因果链（{len(ft_ents)} 个节点, {len(ft_rels)} 条因果）】")
        L.append("说明: gate_type=AND 表示所有下层原因同时成立才触发; gate_type=OR 表示任一原因成立即触发")
        # 找出顶事件：故障树中作为"父节点"出现、且没有上层父节点的顶点
        ft_srcs = set(r["src"] for r in ft_rels)
        ft_tgts = set(r["tgt"] for r in ft_rels)
        top_names = ft_srcs - ft_tgts
        if not top_names:
            # 无法确定顶事件时退化为"父→子"顺序列出
            for r in ft_rels[:25]:
                gate = f"[{r['gate_type'].upper()}]" if r.get("gate_type") else ""
                L.append(f"- {r['src']} {gate} → {r['tgt']}")
        else:
            # 按顶事件分组，自顶向下逐层输出因果链
            for ti, top in enumerate(list(top_names)[:3], 1):
                L.append(f"  故障树{ti}: {top}")
                # BFS 自顶向下组织层级
                layer = [top]
                emitted = set()
                depth = 0
                while layer and depth < 5:
                    depth += 1
                    next_layer = []
                    for parent in layer:
                        # 防止重复边：直接遍历所有 src==parent 的关系
                        for r in ft_rels:
                            if r["src"] != parent:
                                continue
                            gate = f"[{r['gate_type'].upper()}]" if r.get("gate_type") else ""
                            indent = "    " * depth
                            L.append(f"    {indent}{parent} {gate} → {r['tgt']}")
                            if r["tgt"] not in emitted:
                                emitted.add(r["tgt"])
                                next_layer.append(r["tgt"])
                    layer = [n for n in next_layer if n not in emitted][:8]
        # 附上故障树节点的描述信息（帮助 LLM 理解节点含义）
        ft_extra = []
        for e in ft_ents[:15]:
            d = (e.get("desc") or "")
            if d and len(d) > 4:
                ft_extra.append(f"{e['name']}: {d[:60]}")
        if ft_extra:
            L.append(f"  节点说明({len(ft_extra)}): " + "；".join(ft_extra))

    # ── 社区报告（优先级次之，放前面）──
    if reports:
        L.append(f"\n【社区报告（{len(reports)} 个）】")
        for r in reports:
            rating_str = f" (评分:{r.get('rating')})" if r.get("rating") is not None else ""
            L.append(f"- {r['title']}{rating_str}")
            if r.get("summary"):
                L.append(f"  摘要: {r['summary'][:250]}")
            # 社区包含的实体列表（帮助 LLM 理解社区边界）
            r_ents = r.get("entities", []) or []
            if r_ents:
                L.append(f"  社区实体({len(r_ents)}): {', '.join(r_ents[:10])}{'...' if len(r_ents) > 10 else ''}")
            # 关键发现（findings）—— 社区报告的核心价值
            r_findings = r.get("findings", []) or []
            if r_findings:
                L.append(f"  关键发现({len(r_findings)}):")
                for fi, f in enumerate(r_findings[:3], 1):
                    fsum = f.get("summary", "") if isinstance(f, dict) else str(f)
                    L.append(f"    {fi}. {fsum[:150]}")

    # ── 相关实体（细节，限制长度避免挤掉社区报告）──
    if ents:
        L.append(f"\n【相关实体（{len(ents)} 个）】")
        for e in ents[:8]:  # 最多 8 个
            t = f"[{e['type']}]" if e.get("type") else ""
            w = f"(权重:{e['weight']})" if e.get("weight") else ""
            ft_mark = "【故障树】" if e.get("source_type") == "fault_tree" else ""
            desc = (e.get("desc") or "")[:80]  # 限制描述长度
            L.append(f"- {e['name']} {t} {w} {ft_mark}: {desc}")

    # ── 关系（细节，限制长度）──
    if rels:
        L.append(f"\n【关系（{len(rels)} 条）】")
        for r in rels[:10]:  # 最多 10 条
            gate = f"[{r['gate_type'].upper()}]" if r.get("gate_type") else ""
            desc = (r.get("desc") or "")[:80]  # 限制描述长度
            L.append(f"- {r['src']} {gate} -> {r['tgt']}: {desc}")

    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', "\n".join(L))


def _kg_search(kb_ids, query: str, entity_keywords=None, type_keywords=None, top_k: int = 5) -> dict:
    """KG 检索完整链路：实体匹配→故障树展开→多跳BFS→社区报告→融合输出"""
    if isinstance(kb_ids, str):
        kb_ids = [kb_ids]
    kb_ids = [k for k in (kb_ids or []) if k]
    if not kb_ids or not query.strip():
        return {"summary": ""}
    ek = entity_keywords or []
    tk_kws = type_keywords or []
    tk = top_k

    all_ents = _hg_vertices(kb_ids)
    seed_ents = _match_entities(all_ents, query.strip(), ek, tk_kws, tk)

    # ── 故障树因果链：种子命中故障树顶点时，沿 source_type=fault_tree 边
    #    展开完整因果链（顶事件→中间事件→底事件），满足诊断类问题 ──
    ft_ents, ft_rels = [], []
    ft_seed_ids = [e["id"] for e in seed_ents if e.get("source_type") == "fault_tree"]
    if ft_seed_ids:
        try:
            ft_ents, ft_rels = _expand_fault_tree(ft_seed_ids, all_ents)
        except Exception:
            ft_ents, ft_rels = [], []

    # ── 普通实体多跳展开（2 跳，可跨组件找到间接原因）──
    neighbors, hop_rels = _expand_neighbors(seed_ents, all_ents, tk)

    # ── 融合：种子 + 故障树 + 邻居 ──
    all_matched = {}
    for e in seed_ents + ft_ents + neighbors:
        all_matched[e["name"]] = e
    ents = list(all_matched.values())[:tk * 3]

    entity_names = [e["name"] for e in ents if e.get("name")]
    reports = _search_reports(kb_ids, entity_names, query.strip(), tk)

    seen, rels = set(), []
    for r in hop_rels:
        k = (r["src"], r["tgt"], r["type"], r.get("gate_type", ""))
        if k not in seen:
            seen.add(k)
            rels.append(r)
    rels = rels[:tk * 2]

    summary = _build_summary(ents, rels, reports, query.strip(), ek, tk_kws, ft_ents, ft_rels)
    # 构建社区报告预览（含 findings 数量、评分、实体数）
    reports_preview = []
    for r in reports[:3]:
        reports_preview.append({
            "title": r.get("title", "") or "未命名社区",
            "summary": (r.get("summary") or "")[:150],
            "rating": r.get("rating"),
            "findings_count": len(r.get("findings", []) or []),
            "findings_preview": [
                (f.get("summary", "") if isinstance(f, dict) else str(f))[:100]
                for f in (r.get("findings", []) or [])[:2]
            ],
            "entity_count": len(r.get("entities", []) or []),
            "entities_preview": (r.get("entities", []) or [])[:5],
            "community_id": r.get("community_id"),
        })
    return {
        "summary": summary,
        "entity_count": len(ents),
        "relation_count": len(rels) + len(ft_rels),
        "report_count": len(reports),
        "fault_tree_count": len(ft_rels),
        "entities_preview": [{"name": e.get("name", ""), "type": e.get("type", ""),
                              "source_type": e.get("source_type", ""),
                              "desc": (e.get("desc") or "")[:100]} for e in ents[:5]],
        "relations_preview": [{"src": r.get("src", ""), "tgt": r.get("tgt", ""),
                               "type": r.get("type", ""),
                               "gate_type": r.get("gate_type", ""),
                               "desc": (r.get("desc") or "")[:100]} for r in rels[:5]],
        "reports_preview": reports_preview,
    }


async def kg_search_node(state) -> Dict:
    """KG 检索节点（复用迁移自 dify_code_node_kg_lite.py 的逻辑）"""
    query = state.get("rewritten_query") or state.get("query", "")
    kb_id = state.get("kb_id", "")
    # 多库关联：kb_ids 优先，无则回退 kb_id 单库
    kb_ids = state.get("kb_ids") or ([kb_id] if kb_id else [])
    entity_keywords = state.get("entity_keywords", [])
    type_keywords = state.get("type_keywords", [])
    # 知识图谱检索返回条数：读取 agent 配置的 kg_top_k
    top_k = int(state.get("kg_top_k", 5) or 5)

    start = time.time()
    try:
        result = _kg_search(
            kb_ids=kb_ids,
            query=query,
            entity_keywords=entity_keywords,
            type_keywords=type_keywords,
            top_k=top_k,
        )
    except Exception as e:
        logger.error(f"kg_search 调用失败: {e}")
        result = {"summary": "", "entity_count": 0, "relation_count": 0, "report_count": 0,
                  "fault_tree_count": 0,
                  "entities_preview": [], "relations_preview": [], "reports_preview": []}

    duration = round(time.time() - start, 2)
    tool_call = {
        "tool": "kg_search",
        "query": query,
        "entities": result.get("entity_count", 0),
        "relations": result.get("relation_count", 0),
        "reports": result.get("report_count", 0),
        "fault_trees": result.get("fault_tree_count", 0),
        "duration": duration,
        "entities_preview": result.get("entities_preview", []),
        "relations_preview": result.get("relations_preview", []),
        "reports_preview": result.get("reports_preview", []),
    }
    return {
        "kg_summary": result.get("summary", ""),
        "tool_calls": [tool_call],
    }
